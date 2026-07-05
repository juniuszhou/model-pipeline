from __future__ import annotations

import argparse

import torch
from transformers import PretrainedConfig

_DTYPE_ALIASES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _parse_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, str):
        if dtype not in _DTYPE_ALIASES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return _DTYPE_ALIASES[dtype]
    return dtype


class LLMTrainingConfig(PretrainedConfig):
    # it is needed for serialization/deserialization
    # required by HuggingFace PreTrainedConfig base class
    model_type = "llm-training"

    def __init__(
        self,
        d_model: int = 256,
        num_hidden_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        rope_theta: float = 10000.0,
        use_moe: bool = False,
        dtype: torch.dtype | str = "bfloat16",
        vocab_size: int = 21128,
        context_length: int = 128,
        batch_size: int = 4,
        max_steps: int = 500,
        learning_rate: float = 3e-4,
        load_workers: int = 0,
        data_path: str = "data/train.jsonl",
        tokenizer_name: str = "bert-base-chinese",
        moe_num: int = 4,
        top_k: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_moe = use_moe
        self.dtype = _parse_dtype(dtype)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.load_workers = load_workers
        self.data_path = data_path
        self.tokenizer_name = tokenizer_name
        self.moe_num = moe_num
        self.top_k = top_k
        self.dropout = dropout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransformerLM")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--load_workers", type=int, default=0)
    parser.add_argument("--moe_num", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-chinese")
    parser.add_argument("--use_moe", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def get_config() -> LLMTrainingConfig:
    args = parse_args()

    print("args: ", args)

    print("args: ", args.use_moe)
    return LLMTrainingConfig(**vars(args))


class BlockConfig:
    def __init__(
        self,
        d_model: int = 256,
        num_hidden_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        rope_theta: float = 10000.0,
        use_moe: bool = False,
        dtype: torch.dtype | str = "bfloat16",
        vocab_size: int = 21128,
        context_length: int = 128,
        batch_size: int = 4,
        max_steps: int = 500,
        learning_rate: float = 3e-4,
        load_workers: int = 0,
        data_path: str = "data/train.jsonl",
        tokenizer_name: str = "bert-base-chinese",
        moe_num: int = 4,
        top_k: int = 1,
        dropout: float = 0.1,
    ):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_moe = use_moe
        self.dtype = _parse_dtype(dtype)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.load_workers = load_workers
        self.data_path = data_path
        self.moe_num = moe_num
        self.top_k = top_k
        self.tokenizer_name = tokenizer_name
        self.dropout = dropout

    def from_llm_config(self, llm_config: LLMTrainingConfig) -> BlockConfig:
        return BlockConfig(
            d_model=llm_config.d_model,
            num_hidden_layers=llm_config.num_hidden_layers,
            num_heads=llm_config.num_heads,
            d_ff=llm_config.d_ff,
            rope_theta=llm_config.rope_theta,
            use_moe=llm_config.use_moe,
            dtype=llm_config.dtype,
        )
