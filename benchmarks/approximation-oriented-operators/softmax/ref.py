import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)
