import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.amax(x, dim=0)
