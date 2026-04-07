import torch

def torch_kernel(x: torch.Tensor, op_type: int = 0) -> torch.Tensor:
    return torch.softmax(x, dim=-1)