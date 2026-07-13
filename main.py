from torch import Tensor

if __name__ == "__main__":
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.min()
    print(b.shape)
