import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return x.min(dim=0).values
