import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = x.max(dim=-1, keepdim=True).values

z = x - y
print(z.shape)
