#!/bin/bash
source .venv/bin/activate
source .env
echo "HF_TOKEN: $HF_TOKEN"
uv run train_dpo.py \
  --model_name gpt2 \
  --output_dir ./model-pipeline-dpo-smoke \
  --num_epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_length 128 \
  --max_prompt_length 64
