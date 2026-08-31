import torch


def torch_kernel(input_tensor: torch.Tensor) -> torch.Tensor:
    return input_tensor.sum()
