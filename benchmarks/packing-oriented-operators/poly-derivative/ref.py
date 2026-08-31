import torch


def torch_kernel(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    p0 = coeffs[0].item()
    p1 = coeffs[1].item()
    p2 = coeffs[2].item()
    p3 = coeffs[3].item()
    p_x = torch.zeros(n)
    dp_x = torch.zeros(n)
    for i in range(n):
        xi = x[i].item()
        x2 = xi * xi
        x3 = x2 * xi
        p_x[i] = p3 * x3 + p2 * x2 + p1 * xi + p0
        dp_x[i] = 3 * p3 * x2 + 2 * p2 * xi + p1
    return torch.stack([p_x, dp_x])
