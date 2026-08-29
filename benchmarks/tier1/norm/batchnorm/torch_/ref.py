import torch
import torch.nn.functional as F


def torch_kernel(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=gamma,
        bias=beta,
        training=False,
        momentum=0.0,
        eps=1e-5,
    )
