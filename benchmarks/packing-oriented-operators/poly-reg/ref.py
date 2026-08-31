import torch


def torch_kernel(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    c4: torch.Tensor,
) -> torch.Tensor:
    n = c0.shape[0]
    output = torch.zeros(n)
    for i in range(n):
        x = c0[i].item()
        output[i] = c1[i].item() + (x * x * c4[i].item() + x * c3[i].item() + c2[i].item())
    return output
