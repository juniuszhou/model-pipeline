import torch
from modelscope import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "Shanghai_AI_Laboratory/internlm2-1_8b-reward",
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "Shanghai_AI_Laboratory/internlm2-1_8b-reward", trust_remote_code=True
)

chat_1 = [
    {"role": "user", "content": "Hello! What's your name?"},
    {
        "role": "assistant",
        "content": "My name is InternLM2! A helpful AI assistant. What can I do for you?",
    },
]
chat_2 = [
    {"role": "user", "content": "Hello! What's your name?"},
    {"role": "assistant", "content": "I have no idea."},
]


# get reward score for a single chat
score1 = model.get_score(tokenizer, chat_1)
score2 = model.get_score(tokenizer, chat_2)
print("score1: ", score1)
print("score2: ", score2)
