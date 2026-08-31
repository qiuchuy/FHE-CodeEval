import torch
import torch.nn.functional as F


def torch_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    C_in = a.shape[0]
    x = a.unsqueeze(0)
    w = b.repeat(1, C_in, 1, 1)
    return F.conv2d(x, w, bias=None, stride=1, padding=1).squeeze(0)
