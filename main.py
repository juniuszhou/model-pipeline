import torch

x = torch.arange(24).reshape(6, 4).float()
print(x.shape)

print(x.unsqueeze(-1).shape)

# y = torch.tensor([[True, False, True], [False, True, False]])

# print(x[y])
