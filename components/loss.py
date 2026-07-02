from __future__ import annotations

import torch
from torch import Tensor


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Computes the cross entropy loss between logits and labels.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes)
        labels: Ground truth class indices of shape (batch_size,)

    Returns:
        Scalar loss value (mean over batch)
    """
    # For numerical stability, subtract max logit from each row
    logits_shifted = logits - torch.max(logits, dim=-1, keepdim=True).values
    # Compute log softmax
    log_probs = logits_shifted - torch.log(
        torch.sum(torch.exp(logits_shifted), dim=-1, keepdim=True)
    )
    # Get the log probabilities of the true labels
    batch_size = logits.shape[0]
    # Create indices for batch dimension
    batch_indices = torch.arange(batch_size, device=logits.device)
    print(log_probs.shape)
    print(batch_indices.shape)
    print(labels.shape)
    # selec tthe log probabilities of the true labels
    correct_log_probs = log_probs[batch_indices, labels]
    # Negative log likelihood loss
    loss = -correct_log_probs
    # Return mean over batch
    return torch.mean(loss)


def mse_loss(logits: Tensor, labels: Tensor):
    return torch.nn.MSELoss()(logits, labels)
