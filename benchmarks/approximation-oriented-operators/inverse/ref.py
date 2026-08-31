import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / x
