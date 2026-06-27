"""Train a small language model with Direct Preference Optimization (DPO).

The script includes a tiny built-in preference dataset so it can be run as a
minimal example. For useful results, pass a larger JSONL file with prompt,
chosen, and rejected fields.
"""

import argparse
import inspect
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

try:
    import dotenv
except ImportError:
    dotenv = None


DEFAULT_PREFERENCE_DATA = [
    {
        "instruction": "Explain machine learning in one short paragraph.",
        "input": "",
        "chosen": (
            "Machine learning is a way for computers to learn patterns from data "
            "so they can make predictions or decisions without being explicitly "
            "programmed for every case."
        ),
        "rejected": "Machine learning is when computers become alive and think like people.",
    },
    {
        "instruction": "Give two practical benefits of exercise.",
        "input": "",
        "chosen": "Exercise can improve heart health and help reduce stress.",
        "rejected": "Exercise only matters for professional athletes.",
    },
    {
        "instruction": "Answer the math question carefully.",
        "input": "What is 12 multiplied by 8?",
        "chosen": "12 multiplied by 8 is 96.",
        "rejected": "12 multiplied by 8 is 86.",
    },
    {
        "instruction": "Rewrite the sentence to be more polite.",
        "input": "Send me the file now.",
        "chosen": "Could you please send me the file when you have a chance?",
        "rejected": "Give me the file immediately.",
    },
]


def format_prompt(instruction, input_text=""):
    """Use the same simple instruction format as the LoRA training example."""
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def normalize_preference_example(item):
    """Accept either prompt/chosen/rejected or instruction/input/chosen/rejected."""
    if {"prompt", "chosen", "rejected"}.issubset(item):
        return {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }

    required_fields = {"instruction", "chosen", "rejected"}
    missing_fields = required_fields - set(item)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Preference example is missing required fields: {missing}")

    return {
        "prompt": format_prompt(item["instruction"], item.get("input", "")),
        "chosen": item["chosen"],
        "rejected": item["rejected"],
    }


def load_preference_dataset(dataset_path=None):
    """Load a DPO preference dataset from JSONL or fall back to built-in examples."""
    if dataset_path:
        data = []
        with open(dataset_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        data = DEFAULT_PREFERENCE_DATA

    examples = [normalize_preference_example(item) for item in data]
    return Dataset.from_list(examples)


def parse_target_modules(value):
    """Convert a comma-separated list into PEFT target modules."""
    return [module.strip() for module in value.split(",") if module.strip()]


def build_dpo_trainer(model, ref_model, tokenizer, training_args, train_dataset, lora_config):
    """Create DPOTrainer while tolerating small TRL API differences."""
    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": lora_config,
    }

    signature = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    return DPOTrainer(**trainer_kwargs)


def build_dpo_config(args):
    """Create DPOConfig while passing only fields supported by the installed TRL."""
    config_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "bf16": False,
        "fp16": torch.cuda.is_available(),
        "use_cpu": not torch.cuda.is_available(),
        "optim": "adamw_torch",
        "report_to": "none",
    }

    signature = inspect.signature(DPOConfig.__init__)
    supported_kwargs = {
        key: value for key, value in config_kwargs.items() if key in signature.parameters
    }
    return DPOConfig(**supported_kwargs)


def main():
    if dotenv:
        dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Train a model with DPO and LoRA")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional JSONL file with prompt/chosen/rejected preference examples",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./dpo_model", help="Output directory"
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
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL strength")
    parser.add_argument("--max_length", type=int, default=256, help="Max total length")
    parser.add_argument(
        "--max_prompt_length", type=int, default=128, help="Max prompt length"
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="c_attn",
        help="Comma-separated LoRA target modules; use c_attn for GPT-2",
    )

    args = parser.parse_args()
    hf_token = os.getenv("HF_TOKEN")

    print("Loading preference dataset...")
    train_dataset = load_preference_dataset(args.dataset_path)
    print(f"Loaded {len(train_dataset)} preference pairs")

    print(f"Loading model and reference model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=hf_token)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, token=hf_token)
    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=parse_target_modules(args.target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = build_dpo_config(args)

    trainer = build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        lora_config=lora_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving DPO LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("DPO training completed!")


if __name__ == "__main__":
    main()
