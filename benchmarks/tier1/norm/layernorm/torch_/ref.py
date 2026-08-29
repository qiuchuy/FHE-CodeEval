import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, x.shape[-1:])
