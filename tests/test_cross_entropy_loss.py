import logging
import random

import torch

from components.loss import cross_entropy_loss


def test_cross_entropy_loss():
    logits = torch.randn(20, 10)
    labels = torch.randint(0, 10, (20,))
    logging.warning("logits shape: %s", logits.shape)
    logging.warning("labels shape: %s", labels.shape)
    loss = cross_entropy_loss(logits, labels)
    logging.warning("loss as single value: %s", loss.item())
    assert loss.shape == ()
    assert loss.item() >= 0


def test_cross_entropy_loss_with_ignore_index():
    logits = torch.randn(20, 10)
    labels = torch.randint(0, 10, (20,))
    ignore_index = -100
    # set some labels to ignore_index at random
    for i in range(20):
        if random.random() < 0.1:
            labels[i] = ignore_index
    loss = cross_entropy_loss(logits, labels, ignore_index)
    assert loss.shape == ()
    assert loss.item() >= 0
