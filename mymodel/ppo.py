"""PPO (Proximal Policy Optimization) for fine-tuning a TransformerLM policy with RL.

Implements the core PPO algorithm used in RLHF:
1. Generate: Use the current policy (actor) to generate response tokens given prompts
2. Score: Compute log-probs under the actor and a frozen reference model; compute rewards
3. Advantage: Compute discounted returns and advantages from the critic's value estimates
4. Update:
   - Actor: clipped surrogate objective + KL penalty from the reference model
   - Critic: MSE between predicted values and discounted returns

The reward function in this demo is a simple length-based heuristic.
In production, replace it with a trained reward model.

References:
  - Schulman et al., "Proximal Policy Optimization Algorithms", 2017
  - Ziegler et al., "Fine-Tuning Language Models from Human Preferences", 2019
"""

from __future__ import annotations

import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from config import LLMTrainingConfig, get_config
except ImportError:
    from mymodel.config import LLMTrainingConfig, get_config
from jaxtyping import Float, Int
from torch import Tensor

try:
    from model import TransformerLM, load_model_safe, save_model_safe
except ImportError:
    from mymodel.model import TransformerLM, load_model_safe, save_model_safe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critic (value network)
# ---------------------------------------------------------------------------


class Critic(TransformerLM):
    """Value network that predicts the expected return for each token position.

    Reuses the backbone of TransformerLM but replaces lm_head with a
    linear projection d_model -> 1, yielding a scalar value per token.
    """

    def __init__(self, config: LLMTrainingConfig):
        super().__init__(config)
        self.value_head = nn.Linear(config.d_model, 1)
        del self.lm_head  # critic does not predict vocabulary tokens

    def forward(
        self, input_ids: Int[Tensor, "batch seq"]
    ) -> Float[Tensor, "batch seq"]:
        x: Float[Tensor, "batch seq d_model"] = self.embed(input_ids)
        hidden: Float[Tensor, "batch seq d_model"] = self.model(x)
        values: Float[Tensor, "batch seq 1"] = self.value_head(hidden)
        return values.squeeze(-1)


# ---------------------------------------------------------------------------
# Prompt dataset (instructions only — no labels, model generates its own answers)
# ---------------------------------------------------------------------------


class PPODataset(Dataset):
    """Loads instruction prompts from a JSONL file and tokenizes them.

    Expects the same format as ``data/sample_data.jsonl``:
    ``{"instruction": "...", "input": "...", "output": "..."}``

    Only the instruction prompt is kept; the reference output is unused
    because the actor generates its own completions during rollouts.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 64,
        prompt_prefix: str = "### Instruction:\n",
        prompt_sep: str = "\n\n### Input:\n",
        prompt_suffix: str = "\n\n### Response:\n",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = prompt_prefix
        self.sep = prompt_sep
        self.suffix = prompt_suffix

        self.samples: list[str] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                if inp:
                    prompt = f"{self.prefix}{instruction}{self.sep}{inp}{self.suffix}"
                else:
                    prompt = f"{self.prefix}{instruction}{self.suffix}"
                self.samples.append(prompt)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Int[Tensor, "seq"]:
        tokens: Int[Tensor, "seq"] = self.tokenizer(
            self.samples[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        return tokens


# ---------------------------------------------------------------------------
# PPO algorithm
# ---------------------------------------------------------------------------


class PPOTrainer:
    """PPO trainer for fine-tuning a ``TransformerLM`` policy.

    Algorithm summary per iteration:

    **Rollout (no_grad)**:
    1. Greedy-decode response tokens from the actor given prompts
    2. Log-probabilities of those tokens under the actor (old) and a frozen ref model
    3. Reward each response (length-based heuristic for this demo)
    4. Critic values for each token → discounted returns → advantages

    **Update (``ppo_epochs`` inner loops)**:
    5. Clipped surrogate objective (eq.7 of the PPO paper)
    6. KL penalty against the reference model
    7. Value-function MSE (with clipping)

    Args:
        actor: Policy network to optimise.
        ref: Frozen reference policy (same architecture as actor).
        critic: Value network.
        config: Training configuration.
        clip_epsilon: PPO clipping range ε.
        kl_coef: Coefficient for the KL penalty term.
        value_coef: Coefficient for the value loss.
        gamma: Discount factor.
        response_length: Number of tokens to generate per prompt.
        ppo_epochs: Inner PPO update epochs per rollout batch.
        target_kl: Early-stopping threshold for approx KL.
    """

    def __init__(
        self,
        actor: TransformerLM,
        ref: TransformerLM,
        critic: Critic,
        config: LLMTrainingConfig,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.02,
        value_coef: float = 0.5,
        gamma: float = 0.99,
        response_length: int = 32,
        ppo_epochs: int = 4,
        target_kl: float = 0.01,
    ):
        if torch.cuda.is_available():
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cpu")

        self.actor = actor.to(self.device)
        self.ref = ref.to(self.device)
        self.critic = critic.to(self.device)
        self.config = config

        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.value_coef = value_coef
        self.gamma = gamma
        self.response_length = response_length
        self.ppo_epochs = ppo_epochs
        self.target_kl = target_kl

        self.actor_optim = AdamW(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optim = AdamW(self.critic.parameters(), lr=config.learning_rate)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _gather_log_probs(
        logits: Float[Tensor, "batch seq vocab"],
        tokens: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch seq"]:
        """Retrieve the log-probability of the *chosen* token at each position."""
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def _generate(
        self, prompts: Int[Tensor, "batch prompt_len"]
    ) -> tuple[Int[Tensor, "batch total"], Int[Tensor, "batch resp"]]:
        """Greedy-decode ``response_length`` tokens per prompt.

        Returns:
            full: [B, prompt_len + response_length] — concatenated prompt and response.
            responses: [B, response_length] — the newly generated tokens only.
        """
        B, plen = prompts.shape
        full = prompts.clone()
        for _ in range(self.response_length):
            logits = self.actor(full)
            nxt = logits[:, -1:, :].argmax(dim=-1)
            full = torch.cat([full, nxt], dim=-1)
        responses = full[:, plen:]
        return full, responses

    @torch.no_grad()
    def _reward_fn(
        self,
        responses: Int[Tensor, "batch resp"],
        pad_token_id: int = 0,
    ) -> Float[Tensor, " batch"]:
        """Length-based reward: closer to ``response_length`` → higher reward.

        ``reward = 1 - |actual_len - target_len| / target_len``   (∈ [0, 1])

        This is a toy reward for demonstration.  Replace with a trained
        reward model in a real RLHF pipeline.
        """
        lengths = (responses != pad_token_id).sum(dim=-1).float()
        return 1.0 - torch.abs(lengths - self.response_length) / self.response_length

    @torch.no_grad()
    def _compute_advantages(
        self,
        values: Float[Tensor, "batch resp"],
        reward: Float[Tensor, " batch"],
    ) -> tuple[Float[Tensor, "batch resp"], Float[Tensor, "batch resp"]]:
        """Discount the scalar reward backwards to get per-token returns.

        The scalar reward is assigned to the **last** token only.  Earlier
        positions get the discounted version:

            R_t = γ^{(T - t)} * reward_T

        Advantage  A_t = R_t - V(s_t).

        Args:
            values: Critic predictions  [B, response_length].
            reward: Scalar per-sequence reward  [B].

        Returns:
            returns: Discounted returns  [B, response_length].
            advantages: R - V  [B, response_length].
        """
        T = values.shape[1]
        returns = torch.zeros_like(values)
        running = reward
        for t in reversed(range(T)):
            returns[:, t] = running
            running = self.gamma * running
        advantages = returns - values
        return returns, advantages

    # -- single training step ------------------------------------------------

    def train_step(self, prompts: Int[Tensor, "batch prompt_len"]) -> dict[str, float]:
        """One PPO iteration: roll out, compute advantages, update actor & critic.

        Returns a dict of mean metrics for logging.
        """
        B, plen = prompts.shape
        self.actor.train()
        self.critic.train()
        self.ref.eval()

        # ── 1. Rollout ────────────────────────────────────────────────────
        with torch.no_grad():
            full, responses = self._generate(prompts)
            # rlen = responses.shape[1]

            act_logits = self.actor(full)
            ref_logits = self.ref(full)

            # Align logits with response tokens:
            #   logits[:, plen-1, :] predicts responses[:, 0]
            #   logits[:, plen-1:-1, :] covers all response-length positions
            resp_logits = act_logits[:, plen - 1 : -1, :]
            ref_resp_logits = ref_logits[:, plen - 1 : -1, :]

            old_logp = self._gather_log_probs(resp_logits, responses)
            ref_logp = self._gather_log_probs(ref_resp_logits, responses)

            values = self.critic(full)[:, plen:]  # [B, rlen]

            reward = self._reward_fn(responses)  # [B]

            returns, advantages = self._compute_advantages(values, reward)

            # Normalise advantages across the batch (reduces variance)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── 2. PPO update (multiple inner epochs) ─────────────────────────
        stats: dict[str, float] = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "kl": 0.0,
            "approx_kl": 0.0,
        }

        for epoch in range(self.ppo_epochs):
            # --- Actor (policy) ---
            cur_logits = self.actor(full)
            cur_resp_logits = cur_logits[:, plen - 1 : -1, :]
            cur_logp = self._gather_log_probs(cur_resp_logits, responses)

            # Importance-sampling ratio:  π_θ(a|s) / π_θ_old(a|s)
            ratio = torch.exp(cur_logp - old_logp)

            # Clipped surrogate objective (PPO paper eq.7):
            #   L^CLIP = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # KL penalty: keep π_θ close to π_ref
            #   D_KL(π_θ || π_ref) ≈ E[log π_θ - log π_ref]
            kl = (cur_logp - ref_logp).mean()
            actor_loss = actor_loss + self.kl_coef * kl

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optim.step()

            # Early stopping: if the policy has already drifted far from the
            # behaviour policy that collected the rollouts, stop updating.
            approx_kl = (cur_logp - old_logp).mean().item()
            if epoch > 0 and approx_kl > self.target_kl * 1.5:
                break

            # --- Critic (value function) ---
            cur_vals = self.critic(full)[:, plen:]
            # Value clipping (similar to the policy clipping for stability)
            v_clipped = values + torch.clamp(
                cur_vals - values, -self.clip_epsilon, self.clip_epsilon
            )
            vf_loss = F.mse_loss(cur_vals, returns, reduction="none")
            vf_loss_clipped = F.mse_loss(v_clipped, returns, reduction="none")
            critic_loss = torch.max(vf_loss, vf_loss_clipped).mean()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optim.step()

            stats["actor_loss"] += actor_loss.item()
            stats["critic_loss"] += critic_loss.item()
            stats["kl"] += kl.item()
            stats["approx_kl"] += approx_kl

        n_epochs = epoch + 1  # adjust for possible early break
        return {k: v / n_epochs for k, v in stats.items()}

    # -- full training loop -------------------------------------------------

    def train(self, num_steps: int, prompt_loader: DataLoader) -> None:
        """Run ``num_steps`` PPO iterations over prompts from ``prompt_loader``."""
        self.actor.train()
        self.critic.train()
        self.ref.eval()

        data_iter = iter(prompt_loader)
        progress = tqdm(range(num_steps), desc="PPO")
        for step in progress:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(prompt_loader)
                batch = next(data_iter)

            prompts = batch.to(self.device)
            metrics = self.train_step(prompts)
            progress.set_postfix(metrics)

            if (step + 1) % 50 == 0:
                save_model_safe(self.actor, "ppo_actor")
                logger.info("Checkpoint saved at step %d", step + 1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    config = get_config()
    device = config.device if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    config.vocab_size = tokenizer.vocab_size

    # ---- Models ----
    logger.info("Initialising models ...")
    try:
        actor = load_model_safe("latest")
        ref = load_model_safe("latest")
        logger.info("Loaded saved model from models/latest/")
    except FileNotFoundError, OSError:
        logger.warning("No saved model found — initialising a fresh TransformerLM")
        actor = TransformerLM(config)
        ref = TransformerLM(config)
        save_model_safe(actor, "latest")

    ref.eval()
    ref.requires_grad_(False)
    critic = Critic(actor.config)

    param_count = sum(p.numel() for p in actor.parameters())
    logger.info("Actor parameters: %d", param_count)

    # ---- Data ----
    dataset = PPODataset(config.data_path, tokenizer, max_length=config.context_length)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    logger.info(
        "Dataset: %d prompts, batch_size=%d, max_steps=%d",
        len(dataset),
        config.batch_size,
        config.max_steps,
    )

    # ---- PPO training ----
    trainer = PPOTrainer(actor, ref, critic, config)
    trainer.train(num_steps=config.max_steps, prompt_loader=loader)

    save_model_safe(actor, "ppo_actor_final")
    logger.info("PPO training complete. Final actor saved to models/ppo_actor_final/")


if __name__ == "__main__":
    main()
