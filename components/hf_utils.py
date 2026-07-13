import os

from huggingface_hub import login
from transformers import AutoModelForCausalLM


def login_huggingface():
    login(token=os.getenv("HF_TOKEN"))


def get_model(model_name: str):
    login_huggingface()
    return AutoModelForCausalLM.from_pretrained(model_name)
