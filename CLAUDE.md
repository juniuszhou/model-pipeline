# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This repository demonstrates LLM model fine-tuning using LoRA (Low-Rank Adaptation) with the Hugging Face Transformers and PEFT libraries. It includes a complete pipeline for:
1. Preparing instruction-following datasets
2. Fine-tuning small language models (like GPT-2) with LoRA
3. Evaluating and comparing base vs. fine-tuned models
4. Running simple agent demonstrations

## Environment Setup

### Prerequisites
- Python 3.8+
- Git
- UV package manager (recommended) or pip

### Setup Commands
```bash
# Clone repository (if not already done)
git clone <repository-url>
cd model-pipeline

# Create virtual environment using UV (recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Alternative: using pip
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
```

## Key Files and Their Purposes

### Core Scripts
- `train_lora.py`: Main training script for LoRA fine-tuning
- `evaluate_model.py`: Model evaluation, comparison, and agent demo script
- `requirements.txt`: List of Python dependencies

### Data
- `data/sample_data.jsonl`: Sample instruction-following dataset in JSONL format
  - Format: {"instruction": "...", "input": "...", "output": "..."}

### Output Directories
- `./lora_model/`: Saved LoRA fine-tuned model (generated after training)
- `.venv/`: Python virtual environment

## Common Development Tasks

### 1. Preparing Datasets
The dataset should be in JSONL format with instruction/input/output fields:
```jsonl
{"instruction": "Explain concept X", "input": "", "output": "Explanation..."}
{"instruction": "Question Y", "input": "Context Z", "output": "Answer..."}
```

### 2. Training a LoRA Model
```bash
source .venv/bin/activate
python train_lora.py \
  --model_name gpt2 \
  --dataset_path data/sample_data.jsonl \
  --output_dir ./lora_model \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-4
```

Key training parameters:
- `--model_name`: Base model (e.g., gpt2, microsoft/DialoGPT-small)
- `--dataset_path`: Path to training data
- `--output_dir`: Where to save the LoRA adapter
- `--num_epochs`: Training epochs
- `--batch_size`: Batch size per GPU
- `--learning_rate`: Learning rate for training

### 3. Evaluating and Comparing Models
```bash
source .venv/bin/activate
python evaluate_model.py \
  --base_model gpt2 \
  --lora_path ./lora_model \
  --demo_agent
```

Options:
- `--demo_agent`: Run simple agent demonstration
- `--compare_only`: Only run model comparison (skip agent demo)
- `--base_model`: Base model name/path
- `--lora_path`: Path to LoRA adapter

### 4. Running Just the Agent Demo
```bash
source .venv/bin/activate
python evaluate_model.py --demo_agent
```
This will use the fine-tuned model if available, otherwise fall back to the base model.

## Model Architecture and Approach

### LoRA (Low-Rank Adaptation)
- Freezes pretrained model weights
- Injects trainable rank-decomposition matrices into each Transformer block
- Significantly reduces trainable parameters (typically <1% of total)
- Enables efficient fine-tuning on consumer hardware

### Target Modules
For GPT-2 family models, we target the `c_attn` (combined attention) projections:
- Query, key, and value projections in self-attention
- This approach efficiently adapts the model's attention mechanisms

### Training Configuration
- Uses 16-bit precision (fp16) for memory efficiency
- Gradient accumulation to simulate larger batch sizes
- LoRA hyperparameters: r=8 (rank), α=16 (alpha), dropout=0.1

## Evaluation Approach

### Comparison Metrics
The evaluation script compares:
1. **Base Model**: Original pretrained model
2. **Fine-tuned Model**: Base model + LoRA adapter

### Test Prompts
Standardized prompts covering:
- Concept explanation (machine learning)
- Factual questions (capital cities)
- Lists (benefits of exercise)
- Creative writing (renewable energy paragraph)
- Mathematical reasoning (multiplication)

### Simple Agent Demonstration
Shows practical usage by having the model:
- Answer factual questions
- Explain concepts
- Follow multi-step instructions
- Demonstrate instruction-following capability

## Troubleshooting

### Common Issues
1. **CUDA/GPU Issues**: If no GPU is available, training will fall back to CPU (slower but functional)
2. **Tokenization Warnings**: Messages about pad_token are expected with GPT-2 family models
3. **HF Hub Rate Limits**: Unauthenticated requests may be rate-limited; set HF_TOKEN for higher limits
4. **Memory Issues**: Reduce batch_size or sequence length if encountering OOM errors

### Performance Notes
- Training time varies significantly based on hardware:
  - GPU (e.g., RTX 3060): Minutes for small models
  - CPU only: Hours for same task
- Model outputs may vary between runs due to sampling randomness
- LoRA adapters are lightweight (typically 1-10MB) compared to full models (hundreds of MBs)

## Next Steps for Extension
1. **Dataset Expansion**: Add more diverse examples to `sample_data.jsonl`
2. **Model Variations**: Try different base models (e.g., DialoGPT, TinyLlama)
3. **Hyperparameter Tuning**: Experiment with LoRA rank (r), alpha, learning rates
4. **Evaluation Metrics**: Add quantitative metrics (BLEU, ROUGE) alongside qualitative comparison
5. **Deployment**: Export full merged model for deployment with `model.merge_and_unload()`

## Git Best Practices
- Commit dataset changes and configuration modifications
- The `.gitignore` file excludes:
  - Model outputs (`lora_model/`)
  - Virtual environments (`.venv/`)
  - Cache directories (`__pycache__/`, `.cache/`)
  - Log files and temporary outputs

## References
- LoRA: https://arxiv.org/abs/2106.09685
- PEFT Library: https://github.com/huggingface/peft
- Hugging Face Transformers: https://github.com/huggingface/transformers