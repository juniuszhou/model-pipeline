from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

TEXT_KEYS = ("text", "content", "input", "prompt", "sentence")
LABEL_KEYS = ("label", "labels", "target", "category", "class")


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PretrainDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")
        print("len(self.samples): ", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # get sample from dataset, its lenght is unknown. so we may truncate the data or pad the data to the max length.
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            # exclude bos and eos token
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        prefix = (
            [self.tokenizer.bos_token_id]
            if self.tokenizer.bos_token_id is not None
            else []
        )
        suffix = (
            [self.tokenizer.eos_token_id]
            if self.tokenizer.eos_token_id is not None
            else []
        )
        # add bos and eos token
        tokens = prefix + tokens + suffix

        # pad the tokens to the max length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # set all pad_token_id to -100
        # -100 是 PyTorch / Hugging Face 里常用的 ignore_index
        labels[input_ids == self.tokenizer.pad_token_id] = (
            torch.nn.CrossEntropyLoss().ignore_index
        )
        return input_ids, labels


class PretrainDataLoader(DataLoader):
    def __init__(
        self,
        dataset: PretrainDataset,
        batch_size: int = 16,
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )


# def main():
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     dataset_path = "data/train.jsonl"
#     dataset = PretrainDataset(dataset_path, tokenizer)
#     dataloader = PretrainDataLoader(dataset, batch_size=16, num_workers=0)
#     embed = nn.Embedding(tokenizer.vocab_size, 1024)
#     index = 0
#     for batch in dataloader:
#         input_ids, labels = batch
#         x = embed(input_ids)
#         print("x: ", x.shape)
#         index += 1
#         if index > 10:
#             break


# if __name__ == "__main__":
#     main()
