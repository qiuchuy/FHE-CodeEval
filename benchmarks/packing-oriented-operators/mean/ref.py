import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=0)
