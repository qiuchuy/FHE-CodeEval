import torch

def torch_kernel(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    return torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
