import torch
import torch.nn as nn


class ReLU(nn.Module):
    _fhe_profile_nonlinear = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class SumReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(torch.sum(x))


def create_model() -> nn.Module:
    return SumReLU().eval()


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)
