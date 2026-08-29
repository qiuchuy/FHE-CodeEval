import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.amin(x, dim=0)
