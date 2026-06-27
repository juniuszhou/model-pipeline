#!/bin/bash
source .venv/bin/activate
source .env
echo "HF_TOKEN: $HF_TOKEN"
uv run train_lora.py \
  --model_name gpt2 \
  --dataset_path data/sample_data.jsonl \
  --output_dir ./lora_model \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-4
