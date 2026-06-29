from __future__ import annotations

import torch
from torch.optim import Optimizer


class Adam(Optimizer):
    def __init__(self, model, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [torch.zeros_like(p) for p in model.parameters()]
        self.v = [torch.zeros_like(p) for p in model.parameters()]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.model.parameters()):
            if p.grad is None:
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()
