import torch

def torch_kernel(a: torch.Tensor, b: torch.Tensor, op_type: int = 0) -> torch.Tensor:
    return (a * b).sum()