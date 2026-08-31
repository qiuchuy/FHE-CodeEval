import torch


def torch_kernel(x: torch.Tensor) -> torch.Tensor:
    """Batch normalisation computed from the batch itself (no learned params)."""
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-5)
