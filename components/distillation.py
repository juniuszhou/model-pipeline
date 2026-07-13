from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    vocab_size = student_logits.size(-1)

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs.view(-1, vocab_size),
        teacher_probs.view(-1, vocab_size),
        reduction=reduction,
    )
    return (temperature**2) * kl
