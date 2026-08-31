import torch
import torch.nn as nn


class Square(nn.Module):
    _fhe_profile_nonlinear = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class LinearSquare(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)
        self.square = Square()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.square(self.linear(x))


def create_model() -> nn.Module:
    return LinearSquare().eval()


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)
