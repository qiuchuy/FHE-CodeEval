import torch


def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.shape[0]
    res = 0.0
    for i in range(n):
        res += a[i].item() + b[i].item() - 2 * a[i].item() * b[i].item()
    return torch.tensor(res)
