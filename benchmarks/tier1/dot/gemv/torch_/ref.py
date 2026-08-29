import torch


def torch_kernel(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, x)
