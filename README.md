# LLM Model Post-Tuning Pipeline

This repository demonstrates a complete pipeline for post-training Large Language Models using LoRA (Low-Rank Adaptation) and DPO (Direct Preference Optimization) with Hugging Face Transformers, PEFT, and TRL.

## Overview

The project provides:
1. Instruction-following dataset preparation
2. LoRA-based fine-tuning of small language models (GPT-2)
3. DPO preference optimization from chosen/rejected responses
4. Model evaluation and comparison (base vs. fine-tuned)
5. Simple agent demonstration showcasing instruction-following capabilities

## Repository Structure

```
model-pipeline/
├── data/
│   └── sample_data.jsonl          # Instruction-following dataset
├── lora_model/                    # Saved LoRA adapter (after training)
├── dpo_model/                     # Saved DPO LoRA adapter (after DPO training)
├── .venv/                         # Python virtual environment
├── .claude/                       # Claude Code settings
├── .gitignore                     # Git ignore rules
├── CLAUDE.md                      # Claude Code guidance
├__requirements.txt                # Python dependencies
├── README.md                      # This file
├── train_lora.py                  # LoRA fine-tuning script
├── train_dpo.py                   # DPO preference optimization script
└── evaluate_model.py              # Model evaluation, comparison & agent demo
```

## Quick Start

### 1. Environment Setup
```bash
# Using UV (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Alternative: using pip
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
```

### 2. Train a LoRA Model
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

### 3. Train a DPO Preference Model
```bash
source .venv/bin/activate
python train_dpo.py \
  --model_name gpt2 \
  --output_dir ./dpo_model \
  --num_epochs 1 \
  --batch_size 1
```

`train_dpo.py` includes a tiny built-in preference dataset so the command runs immediately. For meaningful behavior, pass a larger JSONL file with `--dataset_path`.

### 4. Evaluate and Compare Models
```bash
source .venv/bin/activate
python evaluate_model.py \
  --base_model gpt2 \
  --lora_path ./lora_model \
  --demo_agent
```

## Detailed Process

### 1. Dataset Preparation

The sample dataset (`data/sample_data.jsonl`) contains instruction-following examples in JSONL format:
```jsonl
{"instruction": "Explain the concept of machine learning in simple terms.", "input": "", "output": "Machine learning is..."}
{"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."}
...
```

### 2. LoRA Fine-Tuning

The `train_lora.py` script implements:
- **Base Model**: GPT-2 (124M parameters)
- **LoRA Configuration**: 
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0.1
  - Target Modules: `c_attn` (combined attention projections)
- **Training Setup**:
  - FP16 precision for memory efficiency
  - Gradient accumulation (effective batch size = batch_size × 4)
  - Learning rate: 1e-4
  - Epochs: Configurable (default: 3)

**Key Benefit**: Only ~0.24% of parameters are updated during training (~295K trainable vs 124M total), making fine-tuning extremely efficient.

### 3. Model Evaluation & Comparison

The `evaluate_model.py` script performs:
- **Model Loading**: Loads both base GPT-2 and the LoRA-adapted model
- **Prompt Testing**: Compares responses on standardized prompts covering:
  - Concept explanations
  - Factual questions
  - List generation
  - Creative writing
  - Mathematical reasoning
- **Simple Agent Demo**: Demonstrates instruction-following capability with predefined questions

### 4. DPO Preference Optimization

The `train_dpo.py` script implements a minimal Direct Preference Optimization loop:
- **Policy model**: The model being trained. In this repo it is the base model plus a LoRA adapter.
- **Reference model**: A frozen copy of the original base model. DPO compares the policy against this model so training improves preferences without drifting too far from the starting model.
- **Preference data**: Each example contains a prompt, a preferred answer (`chosen`), and a worse answer (`rejected`).
- **DPO loss**: Increases the probability of the chosen answer relative to the rejected answer, while using the reference model as a regularizer.

The built-in preference examples are intentionally tiny and only prove the pipeline works. A real DPO run needs hundreds or thousands of high-quality preference pairs.

DPO JSONL format:
```jsonl
{"prompt": "### Instruction:\nExplain LoRA simply.\n\n### Response:\n", "chosen": "LoRA trains small adapter matrices while keeping the base model frozen.", "rejected": "LoRA means the model searches the internet."}
{"instruction": "Answer carefully.", "input": "What is 12 * 8?", "chosen": "12 * 8 is 96.", "rejected": "12 * 8 is 86."}
```

### 5. Expected Outcomes

After training, you should observe:
- The fine-tuned model follows instruction formats better than the base model
- Improved coherence on tasks similar to the training data
- The LoRA adapter is lightweight (~1-2MB) compared to the full model (~500MB)
- Base model may show more repetitive or less focused responses

## Technical Details

### LoRA (Low-Rank Adaptation)
Instead of fine-tuning all model parameters, LoRA:
1. Freezes the pretrained model weights
2. Injects trainable rank-decomposition matrices into each Transformer layer
3. Significantly reduces computational cost and storage requirements
4. Enables rapid experimentation with different adaptations

### DPO (Direct Preference Optimization)
DPO is a preference-tuning method used after supervised fine-tuning or instruction tuning. Instead of training a reward model and then running reinforcement learning, DPO directly optimizes the language model on paired responses:
1. Start with a prompt and two answers: `chosen` and `rejected`
2. Score both answers under the trainable policy model
3. Score both answers under the frozen reference model
4. Update the policy so the chosen answer becomes more likely than the rejected answer
5. Use `beta` to control how strongly the policy is allowed to move away from the reference model

In this project, DPO is combined with LoRA so only the adapter weights are updated. This keeps training lightweight and saves a small adapter in `dpo_model/`.

### Target Modules Selection
For GPT-2 family models, we target `c_attn` (combined attention) which includes:
- Query projection
- Key projection  
- Value projection
This efficiently adapts the model's attention mechanisms, which are crucial for understanding and generating text.

### Training Configuration
- **Precision**: FP16 (mixed precision) for reduced memory usage
- **Batch Training**: Gradient accumulation to simulate larger batches
- **Regularization**: LoRA dropout to prevent overfitting
- **Optimization**: AdamW optimizer with learning rate scheduling

## Customization Options

### Changing the Base Model
Try different models by changing the `--model_name` parameter:
```bash
# Small models good for CPU training
python train_lora.py --model_name sshleef/tinyllama-10m

# Dialogue-focused models  
python train_lora.py --model_name microsoft/DialoGPT-small

# Even smaller for quick experiments
python train_lora.py --model_name distilgpt2
```

The DPO script supports the same idea:
```bash
python train_dpo.py --model_name distilgpt2 --output_dir ./dpo_model
```

### Adjusting LoRA Parameters
Modify the LoRA configuration in `train_lora.py`:
- Higher rank (r): More expressive but more parameters to train
- Lower rank: Faster training but less adaptation capacity
- Alpha: Scaling factor for LoRA updates
- Dropout: Regularization parameter

### Dataset Format
To use your own dataset, create a JSONL file with:
```jsonl
{"instruction": "Your instruction here", "input": "Optional input context", "output": "Expected output"}
```

For DPO, create JSONL with either `prompt/chosen/rejected` or `instruction/input/chosen/rejected`:
```jsonl
{"prompt": "Your prompt here", "chosen": "Preferred response", "rejected": "Less helpful response"}
{"instruction": "Your instruction here", "input": "Optional context", "chosen": "Preferred response", "rejected": "Less helpful response"}
```

## Troubleshooting

### Common Issues

1. **CUDA not available**:
   - Training will fall back to CPU (much slower but functional)
   - Consider reducing batch_size if encountering memory issues

2. **Hugging Face Hub rate limits**:
   - Set HF_TOKEN environment variable for higher limits
   - Export HF_TOKEN=your_token_here before running scripts

3. **Out of Memory (OOM) errors**:
   - Reduce `--batch_size` parameter
   - Reduce sequence length in tokenization (max_length parameter)
   - Use CPU training instead of GPU

4. **Module import errors**:
   - Ensure all dependencies are installed: `uv pip install -r requirements.txt`
   - Restart the shell after installing packages

## Files Explained

### Core Scripts
- `train_lora.py`: Main LoRA fine-tuning implementation
- `train_dpo.py`: DPO preference optimization with LoRA adapters
- `evaluate_model.py`: Model loading, comparison, and agent demonstration

### Data
- `data/sample_data.jsonl`: Training examples in Alpaca-format JSONL
- Can be replaced with larger, more diverse datasets for better results

### Outputs
- `lora_model/adapter_model.safetensors`: LoRA weights (safetensors format)
- `lora_model/adapter_config.json`: LoRA configuration
- `lora_model/tokenizer.*`: Tokenizer files
- `lora_model/training_args.bin`: Training arguments used
- `dpo_model/adapter_model.safetensors`: DPO-trained LoRA weights
- `dpo_model/adapter_config.json`: DPO LoRA configuration

### Configuration
- `requirements.txt`: Exact package versions for reproducibility
- `.gitignore`: Excludes large files and directories from version control
- `CLAUDE.md`: Guidance for future Claude Code sessions

## Results Interpretation

When comparing base vs. fine-tuned models:
- Look for better adherence to instruction formats in the fine-tuned model
- Check for improved coherence and relevance in responses
- Notice whether the fine-tuned model shows patterns from the training data
- Remember that with small datasets and limited training, improvements may be subtle

## Next Steps for Extension

1. **Dataset Enhancement**:
   - Add more diverse, high-quality examples
   - Include edge cases and varied instruction types
   - Balance different task types (QA, summarization, generation, etc.)

2. **Model Variations**:
   - Experiment with different base model sizes and architectures
   - Try newer models like TinyLlama, Phi, or Mistral variants
   - Test with instruction-tuned base models

3. **Training Improvements**:
   - Increase number of epochs for better convergence
   - Tune LoRA hyperparameters (rank, alpha, dropout)
   - Collect preference pairs and tune DPO `beta`
   - Experiment with different learning rate schedules
   - Add validation set for early stopping

4. **Advanced Evaluation**:
   - Add quantitative metrics (BLEU, ROUGE, perplexity)
   - Implement human evaluation for qualitative assessment
   - Test on held-out evaluation datasets
   - Measure specific capabilities (reasoning, creativity, instruction following)

5. **Deployment Preparation**:
   - Merge LoRA weights with base model for easier deployment
   - Convert to different formats (GGUF for llama.cpp, ONNX, etc.)
   - Create API endpoint for model serving
   - Build simple web interface for interaction

## References

- LoRA Paper: https://arxiv.org/abs/2106.09685
- DPO Paper: https://arxiv.org/abs/2305.18290
- PEFT Library: https://github.com/huggingface/peft
- Hugging Face Transformers: https://github.com/huggingface/transformers
- TRL Library: https://github.com/huggingface/trl
- Alpaca Dataset Format: https://github.com/tatsu-lab/stanford_alpaca

## License

This project is for educational and demonstration purposes. Please respect the licenses of any base models or datasets used.