import torch

from components.softmax import softmax


def test_softmax_basic_1d():
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.tensor([0.0900, 0.2447, 0.6652])
    assert torch.allclose(softmax(x), expected, atol=1e-4)


def test_softmax_single_element():
    x = torch.tensor([5.0])
    assert torch.allclose(softmax(x), torch.tensor([1.0]))


def test_softmax_all_zeros():
    x = torch.tensor([0.0, 0.0, 0.0])
    expected = torch.tensor([1.0 / 3] * 3)
    assert torch.allclose(softmax(x), expected)


def test_softmax_large_values():
    x = torch.tensor([1000.0, 1000.0, 1000.0])
    expected = torch.tensor([1.0 / 3] * 3)
    assert torch.allclose(softmax(x), expected)


def test_softmax_negative_values():
    x = torch.tensor([-1.0, -2.0, -3.0])
    result = softmax(x)
    assert torch.allclose(result.sum(), torch.tensor(1.0))
    assert result[0] > result[1] > result[2]


def test_softmax_2d_dim_last():
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    result = softmax(x, dim=-1)
    expected_row = torch.tensor([0.0900, 0.2447, 0.6652])
    assert result.shape == (2, 3)
    assert torch.allclose(result[0], expected_row, atol=1e-4)
    assert torch.allclose(result[1], expected_row, atol=1e-4)


def test_softmax_2d_dim_zero():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = softmax(x, dim=0)
    assert result.shape == (2, 3)
    for j in range(3):
        assert torch.allclose(result[:, j].sum(), torch.tensor(1.0))


def test_softmax_2d_dim_one():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = softmax(x, dim=1)
    assert result.shape == (2, 3)
    for i in range(2):
        assert torch.allclose(result[i].sum(), torch.tensor(1.0))


def test_softmax_probabilities_sum_to_one():
    x = torch.randn(4, 8)
    result = softmax(x, dim=-1)
    assert torch.allclose(result.sum(dim=-1), torch.ones(4))
