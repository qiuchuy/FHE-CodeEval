import torch


def torch_kernel(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded_indices = indices.view(1, 1, -1).expand(params.size(0), params.size(1), -1)
    return torch.gather(params, dim=2, index=expanded_indices)
