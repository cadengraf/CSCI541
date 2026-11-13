"""Identity-initialized spatial processing for shared encoder ROI features."""
import torch
from torch import nn


class SpatialReIDAdapter(nn.Module):
    """Add trainable spatial mixing without changing initial embeddings."""

    def __init__(self, channels=256):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False))
        nn.init.zeros_(self.branch[-1].weight)

    def forward(self, features):
        return features + self.branch(features.float())
