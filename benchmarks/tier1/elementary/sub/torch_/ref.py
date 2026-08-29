import torch


def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a - b
