"""Train a small language model with teacher-student distillation.

Knowledge distillation trains a student model to match both the ground-truth
answers and the softer token probabilities produced by a frozen teacher model.
The default models are both GPT-2 so the example runs with the same cached model
used by the rest of this repository. For a real compression run, use a stronger
teacher and a smaller compatible student, for example gpt2 -> distilgpt2.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    import dotenv
except ImportError:
    dotenv = None


def format_example(item):
    """Format instruction examples consistently with the LoRA script."""
    instruction = item["instruction"]
    input_text = item.get("input", "")
    output = item["output"]

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )

    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def load_instruction_dataset(file_path):
    """Load instruction/output JSONL into a Hugging Face Dataset."""
    examples = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                examples.append({"text": format_example(json.loads(line))})

    return Dataset.from_list(examples)


class DistillationTrainer(Trainer):
    """Trainer that blends standard LM loss with teacher-student KL loss."""

    def __init__(
        self,
        *args,
        teacher_model,
        temperature=2.0,
        alpha=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model.eval()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]
        student_outputs = model(**inputs)
        supervised_loss = student_outputs.loss

        with torch.no_grad():
            teacher_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
            }
            teacher_outputs = self.teacher_model(**teacher_inputs)

        distill_loss = self._distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            labels,
        )
        loss = (self.alpha * distill_loss) + ((1.0 - self.alpha) * supervised_loss)

        return (loss, student_outputs) if return_outputs else loss

    def _distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute KL divergence only on tokens that are part of the labels."""
        temperature = self.temperature

        # Causal LMs predict token n+1 from token n, so align logits with labels.
        student_logits = student_logits[:, :-1, :]
        teacher_logits = teacher_logits[:, :-1, :]
        labels = labels[:, 1:]

        valid_tokens = labels.ne(-100)
        if not valid_tokens.any():
            return student_logits.new_tensor(0.0)

        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        per_token_kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)

        return (per_token_kl[valid_tokens].mean()) * (temperature**2)


def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])


def main():
    if dotenv:
        dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Distill a teacher LLM into a student")
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="gpt2",
        help="Teacher model name or path",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="gpt2",
        help="Student model name or path. Try distilgpt2 for a smaller GPT-2 student.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/sample_data.jsonl",
        help="Instruction JSONL dataset with instruction/input/output fields",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distill_model",
        help="Where to save the distilled student model",
    )
    parser.add_argument(
        "--num_epochs", type=float, default=1.0, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Student learning rate"
    )
    parser.add_argument(
        "--max_length", type=int, default=128, help="Maximum tokenized sequence length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Softens teacher probabilities; common values are 2 to 4",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for distillation loss. 0 uses only labels, 1 uses only teacher.",
    )

    args = parser.parse_args()
    hf_token = os.getenv("HF_TOKEN")

    print("Loading dataset...")
    dataset = load_instruction_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} examples")

    print(f"Loading tokenizer from student model: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading teacher model: {args.teacher_model}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        token=hf_token,
    )
    teacher_model.config.pad_token_id = tokenizer.pad_token_id
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    print(f"Loading student model: {args.student_model}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        token=hf_token,
    )
    student_model.config.pad_token_id = tokenizer.pad_token_id

    if teacher_model.config.vocab_size != student_model.config.vocab_size:
        raise ValueError(
            "Teacher and student must use the same vocabulary for logit distillation. "
            "Use compatible model families, such as gpt2 and distilgpt2."
        )

    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        use_cpu=not torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
    )

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        temperature=args.temperature,
        alpha=args.alpha,
    )

    print("Starting distillation training...")
    trainer.train()

    print(f"Saving distilled student model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Distillation completed!")


if __name__ == "__main__":
    main()