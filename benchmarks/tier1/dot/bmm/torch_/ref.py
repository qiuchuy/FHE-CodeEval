import torch


def torch_kernel(A, B):
    return torch.bmm(A, B)
