import torch


def torch_kernel(x: torch.Tensor, y: torch.Tensor, m: float, b: float) -> torch.Tensor:
    n = x.shape[0]
    output = torch.zeros(n)
    for i in range(n):
        output[i] = y[i].item() + (m * x[i].item() + b)
    return output
