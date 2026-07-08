import torch.nn as nn
from config import get_config
from jaxtyping import Float, Int
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
        x: Float[Tensor, "batch seq d_model"] = self.embed(input_ids)
        outputs: Float[Tensor, "batch seq d_model"] = self.model(x)
        value: Float[Tensor, "batch seq 1"] = self.value_head(outputs)
        loss: Float[Tensor, "batch seq"] = value.squeeze(-1)
        return loss


if __name__ == "__main__":
    config = get_config()
    actor_model = load_model_safe("latest").to(config.device)
    ref_model = load_model_safe("latest").to(config.device)
    ref_model = ref_model.eval().requires_grad_(False)

    critic_model = Critic(actor_model.config)

    # rollout_engine = create_rollout_engine(
    #     engine_type=args.rollout_engine,
    #     policy_model=actor_model,
    #     tokenizer=tokenizer,
    #     device=args.device,
    #     autocast_ctx=autocast_ctx,
    #     sglang_base_url=args.sglang_base_url,
    #     sglang_model_path=args.sglang_model_path,
    #     sglang_shared_path=args.sglang_shared_path,
    # )
