import torch
import torch.nn as nn


class LinearReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 32)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


def create_model() -> nn.Module:
    return LinearReLU().eval()


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)
