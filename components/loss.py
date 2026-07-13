from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_loss(
    logits: Tensor, labels: Tensor, ignore_index: int = -100
) -> Tensor:
    """
    Computes the cross entropy loss between logits and labels.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes)
        labels: Ground truth class indices of shape (batch_size,)
        ignore_index: The index to ignore in the loss calculation.
    Returns:
        Scalar loss value (mean over batch)
    """
    # For numerical stability, subtract max logit from each row
    logits_shifted = logits - torch.max(logits, dim=-1, keepdim=True).values
    # Compute log softmax
    log_probs = logits_shifted - torch.log(
        torch.sum(torch.exp(logits_shifted), dim=-1, keepdim=True)
    )
    # Gather log probabilities for labels, masking ignored positions
    mask = labels != ignore_index
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    correct_log_probs = log_probs[batch_indices, labels.clamp(min=0)]
    loss = -correct_log_probs[mask]
    return loss.mean()


def dpo_loss(
    ref_log_probs: Tensor, policy_log_probs: Tensor, mask: Tensor, beta: float
) -> Tensor:
    """
    Computes the DPO loss between ref_log_probs and policy_log_probs.
    It includes the normal reward of PPO and the KL divergence between the policy and the reference.
    Make the DPO loss more stable and easier to train and no critic model is needed.

    Args:
        ref_log_probs: Reference log probabilities of shape (batch_size, seq_len)
        policy_log_probs: Policy log probabilities of shape (batch_size, seq_len)
        mask: Mask of shape (batch_size, seq_len)
        beta: Beta parameter for the DPO loss
    Returns:
        Scalar loss value (mean over batch)
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    """
    # split chosen and rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[: batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2 :]
    chosen_policy_log_probs = policy_log_probs[: batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2 :]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def mse_loss(logits: Tensor, labels: Tensor):
    return torch.nn.MSELoss()(logits, labels)
