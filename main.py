import torch


def small(*args, **kwargs):
    print(args)
    print(kwargs)


if __name__ == "__main__":
    if torch.accelerator.is_available():
        print(torch.accelerator.current_accelerator())
    else:
        print("No accelerator available")
