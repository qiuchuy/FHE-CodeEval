import torch
import torch.nn as nn


class Square(nn.Module):
    _fhe_profile_nonlinear = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class Conv2dSquare(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.square = Square()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.square(self.conv(x))


def create_model() -> nn.Module:
    return Conv2dSquare().eval()


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)
