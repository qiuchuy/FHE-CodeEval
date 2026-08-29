import torch


def torch_kernel(input_tensor: torch.Tensor, op_type: int = 0) -> torch.Tensor:
    if hasattr(torch, "float8_e4m3fn") and input_tensor.dtype == torch.float8_e4m3fn:
        return torch.sum(input_tensor.float())
    if hasattr(torch, "float8_e5m2") and input_tensor.dtype == torch.float8_e5m2:
        return torch.sum(input_tensor.float())
    if input_tensor.dtype in [torch.float16, torch.bfloat16]:
        return torch.sum(input_tensor.float())
    if input_tensor.dtype == torch.int8:
        return torch.sum(input_tensor.to(torch.int32))
    return torch.sum(input_tensor)
