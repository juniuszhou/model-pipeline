import numpy as np
import torch
from config import LLMTrainingConfig, get_config
from loader import (
    DataLoader,
    DPODataset,
    PretrainDataLoader,
    setup_seed,
)
from model import load_model_safe, save_model_safe
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer


def dpo_loss(
    ref_log_probs: Tensor, policy_log_probs: Tensor, mask: Tensor, beta: float
) -> Tensor:
    # mask out non-response positions, sum log probs per sample over response tokens
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[: batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2 :]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2 :]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)
    return log_probs_per_token


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config: LLMTrainingConfig,
        reference_model: nn.Module,
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.ref_model = reference_model
        self.alpha = config.distillation_alpha
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

    def train(self) -> None:
        self.model.train()
        step = 0
        beta = 0.1
        data_iter = iter(self.dataloader)
        while step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            x_chosen = batch["x_chosen"].to(self.config.device)
            x_rejected = batch["x_rejected"].to(self.config.device)
            y_chosen = batch["y_chosen"].to(self.config.device)
            y_rejected = batch["y_rejected"].to(self.config.device)
            mask_chosen = batch["mask_chosen"].to(self.config.device)
            mask_rejected = batch["mask_rejected"].to(self.config.device)

            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

            ref_logits = self.ref_model(x)

            ref_log_probs = logits_to_log_probs(ref_logits, y)

            logits = self.model(x)
            policy_log_probs = logits_to_log_probs(logits, y)

            loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            step += 1

            if step == 1 or step % 50 == 0 or step == self.config.max_steps:
                print(
                    f"step {step}/{self.config.max_steps} "
                    f"loss={loss:.4f} "
                    f"ppl={np.exp(loss.cpu().item()):.2f}"
                )


def main() -> None:
    setup_seed(42)
    config = get_config()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    config.vocab_size = tokenizer.vocab_size

    dataset = DPODataset(
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

    reference_model = load_model_safe("latest")
    reference_model = reference_model.to(config.device)
    reference_model = torch.compile(reference_model)
    reference_model = reference_model.eval().requires_grad_(False)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {param_count:,}")
    print(
        "approx logits memory (MB): "
        f"{config.batch_size * config.context_length * config.vocab_size * 4 / 1e6:.1f}"
    )

    trainer = Trainer(model, dataloader, config, reference_model)
    trainer.train()
    save_model_safe(model, "latest")


if __name__ == "__main__":
    main()
