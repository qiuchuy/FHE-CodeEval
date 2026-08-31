import torch


def torch_kernel(
    v: torch.Tensor, m_0: torch.Tensor, m_1: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(m_1, torch.matmul(m_0, v))
