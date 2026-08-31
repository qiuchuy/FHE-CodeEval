import torch
import torch.nn as nn


class Square(nn.Module):
    _fhe_profile_nonlinear = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class SumSquare(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.square = Square()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.square(torch.sum(x))


def create_model() -> nn.Module:
    return SumSquare().eval()


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)
