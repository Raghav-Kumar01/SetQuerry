import torch


def get_device():
    """
    Select the best available device for PyTorch.

    CUDA GPU -> preferred
    CPU      -> fallback
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")