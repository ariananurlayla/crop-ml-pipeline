"""
Part B & C - PyTorch model zoo.

  * TemporalCNN1D       : [B, C, 12] -> conv blocks -> optional temporal
                          self-attention -> global average pooling
  * CropClassifier      : temporal features + static features -> class logits
                          (4-class, or binary Rice/Non-Rice for Stage 1)
  * CropRegressor       : same backbone -> single node predicting days after
                          planting (Stage 2)
  * FocalLoss           : gamma-weighted cross-entropy for imbalanced classes
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention across the monthly sequence.

    [B, T, C] -> [B, T, C]. Lets the model weight months dynamically - e.g.
    emphasising the flooded transplant window vs the harvest months for
    padi - instead of treating every month equally in the global average
    pool.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} must divide evenly by num_heads {num_heads}"
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, T, C]
        out, _ = self.attn(x, x, x)
        return out


class TemporalCNN1D(nn.Module):
    """Shared backbone: 1D convolutions over the 12-month temporal axis.

    [B, C, 12] -> conv blocks -> optional self-attention -> GAP -> [B, C_last].

    With `use_attention=True` the final MaxPool is skipped so the sequence
    keeps length 3 for the attention layer (12 -> 6 -> 3 instead of -> 1).
    """

    def __init__(self, in_channels: int, conv_channels: List[int],
                 kernel_size: int = 5, dropout: float = 0.2,
                 use_attention: bool = False, num_heads: int = 4):
        super().__init__()
        self.use_attention = use_attention
        layers, cin = [], in_channels
        for i, cout in enumerate(conv_channels):
            layers += [
                nn.Conv1d(cin, cout, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            ]
            # keep a usable sequence length for the attention layer
            if not (use_attention and i == len(conv_channels) - 1):
                layers += [nn.MaxPool1d(2)]
            layers += [nn.Dropout(dropout)]
            cin = cout
        self.conv = nn.Sequential(*layers)
        self.attn = (TemporalSelfAttention(conv_channels[-1], num_heads)
                     if use_attention else None)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        h = self.conv(x_t)                                   # [B, C_last, T']
        if self.attn is not None:
            h = self.attn(h.transpose(1, 2)).transpose(1, 2)  # attention over T'
        return h.mean(dim=2)                                 # [B, C_last]


class CropClassifier(nn.Module):
    """CNN 1D head: backbone features + static features -> class logits.

    ``n_static=0`` is supported (e.g. the CropHarvest source has no static
    vector): the head simply sizes its first Linear from the backbone width
    alone, and concatenating a ``[B, 0]`` static tensor is a no-op.
    """

    def __init__(self, config, in_channels: int, n_static: int,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = TemporalCNN1D(in_channels, config.conv_channels,
                                      config.kernel_size, config.dropout,
                                      config.use_attention, config.num_heads)
        self.head = nn.Sequential(
            nn.Linear(config.conv_channels[-1] + n_static, config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden, num_classes),
        )

    def forward(self, x_t: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.backbone(x_t), x_s], dim=1)
        return self.head(feat)


class CropRegressor(nn.Module):
    """CNN 1D regression head: predicts days_after_planting (HST) directly.

    Replaces the IFRI manual flood-threshold phase mapping (RMSE ~23 days)
    with a supervised learner - the crop-phase thesis innovation (Stage 2).
    """

    def __init__(self, config, in_channels: int, n_static: int):
        super().__init__()
        self.backbone = TemporalCNN1D(in_channels, config.conv_channels,
                                      config.kernel_size, config.dropout,
                                      config.use_attention, config.num_heads)
        self.head = nn.Sequential(
            nn.Linear(config.conv_channels[-1] + n_static, config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden, 1),
        )

    def forward(self, x_t: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.backbone(x_t), x_s], dim=1)
        return self.head(feat)   # [B, 1]


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Down-weights well-classified (majority-class) samples so the model focuses
    on the hard cases - targeted at the Rice/Corn/Sugarcane imbalance.
    ``gamma=0`` reduces to plain cross-entropy. Works for binary and
    multi-class logits alike.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else alpha.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()
