from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def torch_save_load():
    model = SimpleModel()
    torch.save(model.state_dict(), "tmp/model.pth")
    state_dict = torch.load("tmp/model.pth")
    model.load_state_dict(state_dict)
    return model


def huggingface_save_load():
    model = SimpleModel()
    save_file(model.state_dict(), "tmp/model.safetensors")
    state_dict = load_file("tmp/model.safetensors")
    model.load_state_dict(state_dict)
    return model


def tokenizer_save_load():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("tmp/tokenizer")
    path = Path("tmp/tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer


if __name__ == "__main__":
    torch_save_load()
    huggingface_save_load()
    tokenizer_save_load()
