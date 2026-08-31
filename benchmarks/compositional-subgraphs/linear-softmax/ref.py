import torch
import torch.nn as nn


class MatMulSoftmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2))
        return self.softmax(scores)


def create_model() -> nn.Module:
    return MatMulSoftmax().eval()


def torch_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    model: nn.Module,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(q, k)
