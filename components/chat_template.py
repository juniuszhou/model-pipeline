from pathlib import Path

from transformers import PreTrainedTokenizerFast


def demo_chat_template() -> str:
    tokenizer_file = Path(__file__).parent / "tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))

    if tokenizer.chat_template is not None:
        print("chat_template is already set")
        return tokenizer.chat_template

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<s>' if loop.first else '' }}"
        "{{ message['role'] }}: {{ message['content'] }}"
        "{{ '</s>' if loop.last else '\n' }}"
        "{% endfor %}"
    )
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("prompt:\n", prompt)

    tokenized = tokenizer.apply_chat_template(messages, tokenize=True)
    print("tokenized ids:", tokenized)
    return prompt


if __name__ == "__main__":
    demo_chat_template()
