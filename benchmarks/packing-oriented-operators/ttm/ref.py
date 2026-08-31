import torch


def torch_kernel(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.matmul(t, m)
