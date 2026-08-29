import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=0)
