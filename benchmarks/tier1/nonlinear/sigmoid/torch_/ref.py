import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)
