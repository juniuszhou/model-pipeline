import torch

from components.loss import cross_entropy_loss


def test_cross_entropy_loss():
    logits = torch.randn(20, 10)
    labels = torch.randint(0, 10, (20,))
    loss = cross_entropy_loss(logits, labels)
    assert loss.shape == ()
    assert loss.item() >= 0
