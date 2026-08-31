import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleA(nn.Module):
    """CIFAR ResNet option-A shortcut: spatial stride plus zero-padded channels."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, :: self.stride, :: self.stride]
        channels_to_pad = self.out_channels - self.in_channels
        pad_left = channels_to_pad // 2
        pad_right = channels_to_pad - pad_left
        return F.pad(x, (0, 0, 0, 0, pad_left, pad_right), "constant", 0)


class BasicBlock(nn.Module):
    """Basic residual block used by CIFAR ResNet-20."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = nn.ReLU()

        if stride != 1 or in_planes != planes:
            self.shortcut = DownsampleA(in_planes, planes, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10 (3 stages x 3 residual blocks, 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act = nn.ReLU()
        self.layer1 = self._make_layer(16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def create_model() -> nn.Module:
    """Return a ResNet-20 with default initialization."""
    model = ResNet20()
    model.eval()
    return model


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Args:
        x:     Input images, shape (N, 3, 32, 32), float32, range [-1, 1].
        model: ResNet20 instance (in eval mode).
    Returns:
        Logits tensor of shape (N, 10).
    """
    model.eval()
    with torch.no_grad():
        return model(x)
