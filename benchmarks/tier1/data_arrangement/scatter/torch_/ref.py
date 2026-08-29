import torch


def torch_kernel(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    output = x.clone()
    output.scatter_(dim=3, index=indices.to(torch.long), src=x)
    return output
