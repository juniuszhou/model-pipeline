import torch
from config import LLMTrainingConfig, get_config
from loader import (
    DataLoader,
    PretrainDataLoader,
    PretrainDataset,
    setup_seed,
)
from model import load_model_safe, save_model_safe
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


def distillation_loss(
    student_logits, teacher_logits, temperature=1.0, reduction="batchmean"
):
    vocab_size = student_logits.size(-1)

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs.view(-1, vocab_size),
        teacher_probs.view(-1, vocab_size),
        reduction=reduction,
    )
    return (temperature**2) * kl


class Distiller:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: LLMTrainingConfig,
        teacher_model: nn.Module,
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.teacher_model = teacher_model
        self.alpha = config.distillation_alpha
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

    def train(self) -> None:
        self.model.train()
        step = 0
        for batch in self.dataloader:
            if step >= self.config.max_steps:
                break

            input_ids, labels = batch
            input_ids = input_ids.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = self.teacher_model(input_ids).logits

            student_logits = self.model(input_ids)

            vocab_size = student_logits.size(-1)
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            distill_loss = distillation_loss(shift_logits, shift_teacher_logits)

            loss = self.alpha * ce_loss + (1 - self.alpha) * distill_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 50 == 0 or step == self.config.max_steps - 1:
                print(
                    f"ce_loss: {ce_loss.item():.4f}, distill_loss: {distill_loss.item():.4f}"
                )
                print(f"step {step}: loss: {loss.item():.4f}")
            step += 1


def main() -> None:
    setup_seed(42)
    config = get_config()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    config.vocab_size = tokenizer.vocab_size

    dataset = PretrainDataset(
        config.data_path,
        tokenizer,
        max_length=config.context_length,
    )
    dataloader = PretrainDataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.load_workers,
    )

    # load the model from the latest saved by run train.py
    model = load_model_safe("latest")
    model = model.to(config.device)
    model = torch.compile(model)

    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model)
    teacher_model.resize_token_embeddings(config.vocab_size)
    teacher_model = teacher_model.to(config.device)
    teacher_model = torch.compile(teacher_model)
    teacher_model = teacher_model.eval().requires_grad_(False)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {param_count:,}")
    print(
        "approx logits memory (MB): "
        f"{config.batch_size * config.context_length * config.vocab_size * 4 / 1e6:.1f}"
    )

    trainer = Distiller(model, dataloader, config, teacher_model)
    trainer.train()
    save_model_safe(model, "latest")


if __name__ == "__main__":
    main()
