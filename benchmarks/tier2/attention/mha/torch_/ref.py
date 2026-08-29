import torch


def torch_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, N_CTX, H, D_HEAD = q.shape
    q_b = q.view(B * N_CTX, H, D_HEAD)
    k_b = k.view(B * N_CTX, H, D_HEAD)
    v_b = v.view(B * N_CTX, H, D_HEAD)
    scores = torch.bmm(q_b, k_b.transpose(1, 2))
    p = torch.nn.functional.softmax(scores / (D_HEAD**0.5), dim=-1)
    output_b = torch.bmm(p, v_b)
    return output_b.view(B, N_CTX, H, D_HEAD)
