import torch
import torch.nn as nn


class MLP(nn.Module):
    """3-layer MLP for MNIST digit classification (784 -> 256 -> 128 -> 10)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def create_model() -> nn.Module:
    """Return an MLP with default PyTorch initialization (no pretrained weights)."""
    model = MLP()
    model.eval()
    return model


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Args:
        x:     Input tensor, shape (N, 1, 28, 28), float32, range [-1, 1].
        model: MLP instance (in eval mode).
    Returns:
        Logits tensor of shape (N, 10).
    """
    model.eval()
    with torch.no_grad():
        return model(x)
