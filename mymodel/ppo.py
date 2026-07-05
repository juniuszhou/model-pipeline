import torch.nn as nn
from config import get_config
from jaxtyping import Int
from model import LLMTrainingConfig, TransformerLM, load_model_safe
from torch import Tensor


class PPO:
    def __init__(self, actor_model, critic_model, config):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.config = config

    def train(self):
        pass


class Critic(TransformerLM):
    def __init__(self, config: LLMTrainingConfig):
        super().__init__(config)
        self.value_head = nn.Linear(config.d_model, 1)

    def forward(self, input_ids: Int[Tensor, "batch seq"]):
        outputs = self.model(input_ids=input_ids)
        hidden_states = self.model.norm(outputs[0])
        value = self.value_head(hidden_states)
        return value


if __name__ == "__main__":
    actor_model = load_model_safe("latest")
    critic_model = load_model_safe("latest")
    config = get_config()
    ppo = PPO(actor_model, critic_model, config)
    ppo.train()
