"""Test the config module."""

import argparse
import sys

import torch

from projects.config import LLMTrainingConfig, parse_args


def test_config_defaults():
    config = LLMTrainingConfig()
    assert config.hidden_size == 256
    assert config.num_hidden_layers == 4
    assert config.num_heads == 8
    assert config.d_ff == 1024
    assert not config.use_moe
    assert config.dtype == torch.bfloat16


def test_config_from_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=4096)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args([])
    config = LLMTrainingConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_moe=args.use_moe,
        dtype=args.dtype,
    )
    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 8
    assert config.num_heads == 16
    assert config.d_ff == 4096
    assert config.use_moe is False
    assert config.dtype == torch.bfloat16

    args = parser.parse_args([
        "--hidden_size",
        "512",
        "--use_moe",
        "--dtype",
        "float16",
    ])
    config = LLMTrainingConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_moe=args.use_moe,
        dtype=args.dtype,
    )
    assert config.hidden_size == 512
    assert config.use_moe is True
    assert config.dtype == torch.float16


def test_parse_args_ignores_pytest_argv(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["trainer.py", "--hidden_size", "128", "--max_steps", "10"],
    )
    args = parse_args()
    assert args.hidden_size == 128
    assert args.max_steps == 10
