import torch


def torch_kernel(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    x0 = x[0].item()
    x1 = x[1].item()
    x2 = x[2].item()
    x3 = x[3].item()
    p0 = p[0].item()
    p1 = p[1].item()
    p2 = p[2].item()
    p3 = p[3].item()
    s0 = x0 + x3
    d0 = x0 - x3
    s1 = x1 + x2
    d1 = x1 - x2
    C0 = (s0 + s1) * p0
    C1 = d0 * p1 + d1 * p2
    C2 = (s0 - s1) * p3
    C3 = d0 * p2 - d1 * p1
    return torch.tensor([C0, C1, C2, C3])
