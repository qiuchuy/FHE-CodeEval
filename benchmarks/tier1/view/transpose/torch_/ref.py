import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.transpose(x, 0, 1).contiguous()