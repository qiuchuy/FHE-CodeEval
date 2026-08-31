import torch
import torch.nn as nn


class Conv2dReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def create_model() -> nn.Module:
    return Conv2dReLU().eval()


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)
