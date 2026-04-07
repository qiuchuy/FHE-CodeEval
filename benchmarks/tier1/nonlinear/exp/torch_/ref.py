import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)
