"""
a simple evaluation script for the model
"""

import argparse

from lm_eval import simple_evaluate


def evaluate_model(model_name, tasks="lambada", num_fewshot=0):
    """Evaluate a model using Hugging Face lm-eval simple_evaluate API."""
    print(f"Evaluating model: {model_name}")
    print(f"Tasks: {tasks}")
    print("-" * 50)

    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name}",
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        batch_size=8,
        max_batch_size=8,
        log_samples=True,
        verbosity="INFO",
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    # Print overall metric scores (the evaluation matrix)
    for task_name, task_results in results["results"].items():
        print(f"\nTask: {task_name}")
        for metric, value in task_results.items():
            print(f"  {metric}: {value}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Simple LLM evaluation with lm-eval")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument(
        "--tasks",
        type=str,
        default="lambada,truthfulqa_mc1,piqa",
        help="Comma-separated list of evaluation tasks",
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0, help="Number of few-shot examples"
    )

    args = parser.parse_args()

    evaluate_model(args.model, args.tasks, args.num_fewshot)


if __name__ == "__main__":
    main()
