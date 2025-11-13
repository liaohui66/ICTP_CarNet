# models/film.py
import torch.nn as nn
import torch

class GlobalFiLM(nn.Module):
    """Graph-level FiLM to modulate the 7-dim l=3 coefficient vector."""
    def __init__(self, g_in: int, out_dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(g_in, hidden), nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 2 * out_dim)
        )
    def forward(self, gfeat: torch.Tensor):
        gb = self.net(gfeat)             # (B, 2*out_dim)
        return torch.chunk(gb, 2, dim=-1)  # gamma, beta
