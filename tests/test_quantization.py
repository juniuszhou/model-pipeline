import torch

from mymodel.quantization import asymmetric_scale_and_z, symmetric_scale


def test_quantization():
    a = torch.range(-126, 126, 20)
    print(a)
    scale, zero_point = asymmetric_scale_and_z(a, q_min=-128, q_max=127)
    print(scale, zero_point)


def test_quantization_symmetric():
    a = torch.range(-126, 126, 20)
    print(a)
    scale = symmetric_scale(a, q_max=127)
    print(scale)
