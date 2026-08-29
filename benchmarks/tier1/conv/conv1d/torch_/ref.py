import torch


def torch_kernel(input_tensor, kernel_tensor):
    input_expanded = input_tensor.unsqueeze(0).unsqueeze(0)
    kernel_expanded = kernel_tensor.unsqueeze(0).unsqueeze(0)
    result = torch.nn.functional.conv1d(input_expanded, kernel_expanded, padding=0)
    return result.squeeze()
