import torch


def torch_kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(a, b), c)
