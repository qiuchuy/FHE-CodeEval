import torch
import torch.nn as nn


class BertAttention(nn.Module):
    """BERT-style attention core matching the Rotom benchmark structure."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        first_head_q = q[:, 0]
        first_head_k_t = k[:, 0].transpose(-1, -2)
        first_head_v = v[:, 0]

        # Match the Rotom benchmark: first-head (Q @ K.T) @ V without scaling or softmax.
        return torch.matmul(torch.matmul(first_head_q, first_head_k_t), first_head_v)


def create_model() -> nn.Module:
    """Return a BERT-attention block with default PyTorch initialization."""
    model = BertAttention()
    model.eval()
    return model


def torch_kernel(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Args:
        x:     Input embeddings, shape (N, 128, 768), float32, range [0, 1].
        model: BertAttention instance (in eval mode).
    Returns:
        First-head attention output of shape (N, 128, 64).
    """
    model.eval()
    with torch.no_grad():
        return model(x)
