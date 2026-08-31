import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 adapted for CIFAR-10 (32x32 RGB input, 10 classes).

    Uses AvgPool2d (FHE-friendly) and ReLU activations.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=0)   # 32 -> 28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)  # 14 -> 10 (after pool)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv1(x)))   # (N, 6, 14, 14)
        x = self.pool(self.act(self.conv2(x)))   # (N, 16, 5, 5)
        x = x.view(x.size(0), -1)               # (N, 400)
        x = self.act(self.fc1(x))               # (N, 120)
        x = self.act(self.fc2(x))               # (N, 84)
        return self.fc3(x)                       # (N, 10)


def create_model() -> nn.Module:
    """Return a LeNet-5 with default PyTorch initialization (no pretrained weights)."""
    model = LeNet5()
    model.eval()
    return model


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Args:
        x:     Input images, shape (1, 3, 32, 32), float32, range [-1, 1].
        model: LeNet5 instance (in eval mode).
    Returns:
        Logits tensor of shape (1, 10).
    """
    model.eval()
    with torch.no_grad():
        return model(x)
