import numpy as np
import torch
import torch.nn as nn
from config import LLMTrainingConfig, get_config
from loader import PretrainDataLoader, SFTDataset, setup_seed
from lora import apply_lora
from model import load_model_safe
from torch.optim import AdamW
from transformers import AutoTokenizer


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: PretrainDataLoader,
        config: LLMTrainingConfig,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.dataloader = dataloader
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self) -> None:
        self.model.train()
        losses: list[float] = []
        step = 0
        data_iter = iter(self.dataloader)

        # while step < 2:
        while step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            step += 1
            input_ids, labels = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step} "
                    f"(all {shift_labels.numel()} labels may be -100/ignored)"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)
            if step == 1 or step % 50 == 0 or step == self.config.max_steps:
                print(
                    f"step {step}/{self.config.max_steps} "
                    f"loss={loss_value:.4f} "
                    f"ppl={np.exp(loss_value):.2f}"
                )

        if len(losses) >= 2 and losses[-1] >= losses[0]:
            print(
                "Warning: final loss did not decrease versus step 1; "
                "try more steps or adjust learning rate."
            )
        # save_model_safe(self.model, name="latest")


def main() -> None:
    setup_seed(42)
    config = get_config()

    print("config: ", config.use_moe)

    if config.use_moe:
        print("Warning: use_moe is not implemented yet; training dense FFN.")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    config.vocab_size = tokenizer.vocab_size

    dataset = SFTDataset(
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

    model = load_model_safe("latest")
    apply_lora(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {param_count:,}")
    print(
        "approx logits memory (MB): "
        f"{config.batch_size * config.context_length * config.vocab_size * 4 / 1e6:.1f}"
    )

    trainer = Trainer(model, dataloader, config)
    trainer.train()


if __name__ == "__main__":
    main()
