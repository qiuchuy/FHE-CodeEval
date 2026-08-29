import torch


def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat((a, b), dim=1)
