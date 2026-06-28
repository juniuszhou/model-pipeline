#!/bin/bash
set -euo pipefail

# cd "$(dirname "$0")/.."

# Sized for ~6GB GPU: large models OOM mainly on logits (batch * seq * vocab).
uv run trainer.py \
    --hidden_size 256 \
    --num_hidden_layers 4 \
    --num_heads 8 \
    --d_ff 1024 \
    --rope_theta 10000.0 \
    --dtype bfloat16 \
    --context_length 128 \
    --batch_size 4 \
    --max_steps 500 \
    --learning_rate 3e-4 \
    --load_workers 0 \
    --data_path data/train.jsonl \
    --tokenizer_name bert-base-chinese \
    --moe_num 4 \
    --top_k 2 \
    --use_moe True