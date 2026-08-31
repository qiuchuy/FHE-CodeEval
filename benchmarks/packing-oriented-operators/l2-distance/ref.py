import torch


def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.shape[0]
    res = 0.0
    for i in range(n):
        diff = a[i].item() - b[i].item()
        res += diff * diff
    return torch.tensor(res)
