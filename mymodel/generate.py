from __future__ import annotations

from jaxtyping import Int
from model import load_model_safe
from torch import Tensor
from transformers import AutoTokenizer


def main() -> None:
    model = load_model_safe("latest")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    prompt = "Hello, how are you? explain what is the universe"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print("input_ids: ", input_ids.shape)

    output_ids: Int[Tensor, "seq"] = model.generate(input_ids, max_new_tokens=100)
    print("output_ids: ", output_ids.shape)
    # result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
