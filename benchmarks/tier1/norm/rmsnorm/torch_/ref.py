import torch

def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + 1e-6)
