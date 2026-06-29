import torch
import torch.nn as nn

from components.gradients import Adam


def test_adam():
    model = nn.Linear(10, 10)
    optimizer = Adam(model)
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    assert model.weight.grad is not None
    assert model.bias.grad is not None
