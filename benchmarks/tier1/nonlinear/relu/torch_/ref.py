import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)
