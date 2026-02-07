from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=0, bias=True)
        self.left_pad = (k - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


class LayerNormChannel(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> LN over channel dim by transposing to [B, L, C]
        x = x.transpose(1, 2).contiguous()
        x = self.ln(x)
        return x.transpose(1, 2).contiguous()


class ResBlock(nn.Module):
    def __init__(self, C: int, k: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(C, C, k, dilation)
        self.act1 = nn.GELU()
        self.norm1 = LayerNormChannel(C)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(C, C, k, dilation)
        self.act2 = nn.GELU()
        self.norm2 = LayerNormChannel(C)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.drop1(self.norm1(self.act1(self.conv1(x))))
        x = self.drop2(self.norm2(self.act2(self.conv2(x))))
        return x + residual


class DilatedCausalCNN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        C: int = 64,
        k: int = 5,
        dilations: Iterable[int] = (1, 2, 4, 8, 16, 32),
        horizon: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(in_feats, C, kernel_size=1)
        self.blocks = nn.ModuleList([ResBlock(C, k=k, dilation=d, dropout=dropout) for d in dilations])
        self.head_norm = LayerNormChannel(C)
        self.head = nn.Linear(C, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F] -> yhat: [B, H]
        """
        # to [B, F, L] for Conv1d
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        for b in self.blocks:
            x = b(x)
        last_step = self.head_norm(x)[:, :, -1]  # [B, C]
        yhat = self.head(last_step)              # [B, H]
        return yhat


def receptive_field(k: int, dilations: Iterable[int]) -> int:
    """
    Two causal convs per ResBlock; each adds (k - 1) * d to RF.
    RF = 1 + 2 * (k - 1) * sum(dilations)
    """
    return 1 + 2 * (k - 1) * sum(dilations)
