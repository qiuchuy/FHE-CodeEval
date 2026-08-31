import torch
import torch.nn as nn


class Fire(nn.Module):
    """SqueezeNet Fire module: squeeze 1x1, then parallel 1x1 and 3x3 expands."""

    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
    ):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2d(
            squeeze_channels, expand1x1_channels, kernel_size=1
        )
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            dim=1,
        )


class SqueezeNet(nn.Module):
    """
    SqueezeNet 1.1 adapted for CIFAR-10 (32x32 RGB input, 10 classes).

    This variant swaps the original max-pooling layers for average pooling to
    keep the architecture more FHE-friendly. With the fixed 32x32 CIFAR input,
    the classifier head already reaches a 1x1 spatial map, so a fixed AvgPool2d
    is sufficient in the final stage.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def create_model() -> nn.Module:
    """Return a dropout-free SqueezeNet model with default initialization."""
    model = SqueezeNet()
    model.eval()
    return model


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Args:
        x:     Input images, shape (N, 3, 32, 32), float32, range [-1, 1].
        model: SqueezeNet instance (in eval mode).
    Returns:
        Logits tensor of shape (N, 10).
    """
    model.eval()
    with torch.no_grad():
        return model(x)
