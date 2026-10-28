"""Shared ShuffleNetV2 backbone with semantic mask and 128D ReID heads."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ShuffleNet_V2_X1_5_Weights, shufflenet_v2_x1_5

EMBEDDING_SIZE = 128


class JointNet(nn.Module):
    def __init__(self, pretrained: bool, use_mask_pool: bool = True):
        super().__init__()
        self.use_mask_pool = use_mask_pool
        self.backbone = shufflenet_v2_x1_5(
            weights=ShuffleNet_V2_X1_5_Weights.DEFAULT if pretrained else None
        )
        self.mask_head = nn.Sequential(nn.Conv2d(176, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 21, 1))
        self.mask_context = nn.Sequential(nn.Conv2d(352, 64, 1), nn.ReLU(), nn.Conv2d(64, 21, 1))
        self.embed_head = nn.Linear(1024, EMBEDDING_SIZE)

    def forward(self, image):
        """Return VOC logits (or None) and an unnormalized 128D embedding."""
        x = self.backbone.maxpool(self.backbone.conv1(image))
        low = self.backbone.stage2(x)
        mid = self.backbone.stage3(low)
        mask_logits = None
        if getattr(self, "compute_mask", True):
            mask_logits = self.mask_head(low) + F.interpolate(
                self.mask_context(mid), size=low.shape[-2:], mode="bilinear", align_corners=False
            )  # 21 VOC classes, 28 x 28 for 224 x 224 input
        x = self.backbone.conv5(self.backbone.stage4(mid))
        if self.use_mask_pool:
            foreground = F.adaptive_avg_pool2d(1 - mask_logits.softmax(dim=1)[:, :1], x.shape[-2:])
            pooled = (x * foreground).sum((2, 3)) / foreground.sum((2, 3)).clamp_min(1e-6)
        else:
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        embedding = self.embed_head(pooled)
        return mask_logits, embedding


def mask_loss(logits, targets):
    targets = F.interpolate(targets[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
    weights = torch.ones(21, device=logits.device)
    weights[0] = 0.2  # VOC background dominates the pixels otherwise.
    return F.cross_entropy(logits, targets, weight=weights, ignore_index=255)
