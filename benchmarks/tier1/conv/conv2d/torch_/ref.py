import torch


def torch_kernel(input_tensor, kernel_tensor):
    output = torch.nn.functional.conv2d(
        input_tensor, kernel_tensor, stride=3, padding=0
    )
    return output
