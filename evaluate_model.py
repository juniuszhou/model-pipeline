import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model(model_name):
    """Load the base model"""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_finetuned_model(base_model_name, lora_path):
    """Load the base model and apply LoRA weights"""
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    print(f"Loading LoRA weights from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=100):
    """Generate response from model given a prompt"""
    # Format prompt for instruction following (same as training)
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(
        formatted_prompt, return_tensors="pt", truncation=True, max_length=128
    )

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part (after "### Response:\n")
    if "### Response:\n" in response:
        response = response.split("### Response:\n")[1]
    return response.strip()


def compare_models(base_model_name, lora_path, test_prompts):
    """Compare base and fine-tuned models on test prompts"""
    print("=" * 50)
    print("LOADING MODELS")
    print("=" * 50)

    # Load models
    base_model, tokenizer = load_base_model(base_model_name)
    finetuned_model, _ = load_finetuned_model(base_model_name, lora_path)

    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest Prompt {i}: {prompt}")
        print("-" * 50)

        # Generate responses
        base_response = generate_response(base_model, tokenizer, prompt)
        finetuned_response = generate_response(finetuned_model, tokenizer, prompt)

        print(f"Base Model Response:\n{base_response}")
        print()
        print(f"Fine-tuned Model Response:\n{finetuned_response}")
        print()


def simple_agent_demo(model_name, lora_path=None):
    """Demonstrate a very basic agent using the model"""
    print("=" * 50)
    print("SIMPLE AGENT DEMONSTRATION")
    print("=" * 50)

    if lora_path and os.path.exists(lora_path):
        print("Using fine-tuned model for agent")
        model, tokenizer = load_finetuned_model(model_name, lora_path)
    else:
        print("Using base model for agent")
        model, tokenizer = load_base_model(model_name)

    # Agent-like loop: answer a few predefined questions
    agent_questions = [
        "What is the capital of Japan?",
        "Explain why the sky is blue in simple terms.",
        "What are the first three prime numbers?",
        "How do you make a cup of tea?",
        "What is the main purpose of a compass?",
    ]

    for question in agent_questions:
        print(f"\nUser: {question}")
        response = generate_response(model, tokenizer, question, max_length=150)
        print(f"Agent: {response}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare LLM models")
    parser.add_argument(
        "--base_model", type=str, default="gpt2", help="Base model name"
    )
    parser.add_argument(
        "--lora_path", type=str, default="./lora_model", help="Path to LoRA model"
    )
    parser.add_argument(
        "--demo_agent", action="store_true", help="Run simple agent demo"
    )
    parser.add_argument(
        "--compare_only", action="store_true", help="Only run model comparison"
    )

    args = parser.parse_args()

    # Test prompts for comparison
    test_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What is the capital of France?",
        "List three benefits of regular exercise.",
        "Write a short paragraph about the importance of renewable energy.",
        "What is 15 multiplied by 7?",
    ]

    if not args.compare_only:
        # Run agent demo
        simple_agent_demo(
            args.base_model, args.lora_path if os.path.exists(args.lora_path) else None
        )

    # Always run comparison if LoRA model exists
    if os.path.exists(args.lora_path):
        compare_models(args.base_model, args.lora_path, test_prompts)
    else:
        print(f"LoRA model not found at {args.lora_path}. Skipping comparison.")
        print("To create a LoRA model, run: python train_lora.py")


if __name__ == "__main__":
    main()
