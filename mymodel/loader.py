from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Features, Value, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

TEXT_KEYS = ("text", "content", "input", "prompt", "sentence")
LABEL_KEYS = ("label", "labels", "target", "category", "class")

chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get("tools") for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    # 概率性添加system
    if conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


class PretrainDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")
        print("len(self.samples): ", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # get sample from dataset, its lenght is unknown. so we may truncate the data or pad the data to the max length.
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            # exclude bos and eos token
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        prefix = (
            [self.tokenizer.bos_token_id]
            if self.tokenizer.bos_token_id is not None
            else []
        )
        suffix = (
            [self.tokenizer.eos_token_id]
            if self.tokenizer.eos_token_id is not None
            else []
        )
        # add bos and eos token
        tokens = prefix + tokens + suffix

        # pad the tokens to the max length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # set all pad_token_id to -100
        # -100 是 PyTorch / Hugging Face 里常用的 ignore_index
        labels[input_ids == self.tokenizer.pad_token_id] = (
            torch.nn.CrossEntropyLoss().ignore_index
        )
        return input_ids, labels


class PretrainDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        """ data format:
        {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! I am MiniMind. How can I assist you today?"}]}
        """
        features = Features({
            "conversations": [
                {
                    "role": Value("string"),
                    "content": Value("string"),
                    "reasoning_content": Value("string"),
                    "tools": Value("string"),
                    "tool_calls": Value("string"),
                }
            ]
        })
        self.samples = load_dataset(
            "json", data_files=jsonl_path, split="train", features=features
        )
        if tokenizer.chat_template is None:
            tokenizer.chat_template = chat_template

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = (
                    json.loads(message["tools"])
                    if isinstance(message["tools"], str)
                    else message["tools"]
                )
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, tools=tools
            )
        return "\n".join(
            f"{m['role'].capitalize()}: {m.get('content', '')}" for m in messages
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample["conversations"])

        all_input_ids: list[int] = []
        all_labels: list[int] = []

        for msg in conversations:
            role = msg["role"]
            msg_text = self.tokenizer.apply_chat_template(
                [dict(msg)], tokenize=False, add_generation_prompt=False
            )
            tokens = self.tokenizer(msg_text, add_special_tokens=False).input_ids
            all_input_ids.extend(tokens)
            if role == "assistant":
                all_labels.extend(tokens)
            else:
                all_labels.extend([-100] * len(tokens))

        # Truncate to max_length
        input_ids = all_input_ids[: self.max_length]
        labels = all_labels[: self.max_length]

        # Pad
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        valid_count = sum(1 for l in labels if l != -100)
        if valid_count == 0:
            raise RuntimeError(
                f"No valid assistant labels found in sample {index}. "
                f"Increase context_length (currently {self.max_length}) — "
                f"chat template overhead uses all available tokens."
            )

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long
        )


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset_path = "data/train.jsonl"
    dataset = PretrainDataset(dataset_path, tokenizer)
    dataloader = PretrainDataLoader(dataset, batch_size=16, num_workers=0)
    embed = nn.Embedding(tokenizer.vocab_size, 1024)
    index = 0
    for batch in dataloader:
        input_ids, labels = batch
        x = embed(input_ids)
        print("x: ", x.shape)
        index += 1
        if index > 10:
            break


# if __name__ == "__main__":
#     main()
