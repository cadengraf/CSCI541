"""One fixed-input, fixed-output identity model and its image preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from torch import nn


BACKBONE = "efficientnet_b0.ra_in1k"
INPUT_SIZE = 160
EMBEDDING_SIZE = 64


class IdentityEncoder(nn.Module):
    """Accept RGB float pixels in [0, 1]; emit one raw 64-value vector.

    L2 normalization stays on the host so conversion needs only a single
    linear output. The input normalization is inside the exported network.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
        self.projection = nn.Linear(self.backbone.num_features, EMBEDDING_SIZE, bias=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def features(self, rgb01: torch.Tensor) -> torch.Tensor:
        return self.backbone((rgb01 - self.mean) / self.std)

    def forward(self, rgb01: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(rgb01))


def crop_rgb01(image_bgr: np.ndarray, bbox_xywh: list[int]) -> np.ndarray:
    """Match sensor ROI behavior: box crop rescaled directly to 160 square."""
    x, y, w, h = (int(value) for value in bbox_xywh)
    crop = image_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"empty image crop for box {bbox_xywh}")
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def observation_input(row: dict, base_dir: Path) -> np.ndarray:
    image = cv2.imread(str(base_dir / row["image_path"]))
    if image is None:
        raise ValueError(f"cannot read {row['image_path']}")
    return crop_rgb01(image, row["bbox_xywh"])


def load_checkpoint(path: Path) -> IdentityEncoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("backbone") != BACKBONE or
            checkpoint.get("input_size") != INPUT_SIZE or
            checkpoint.get("embedding_size") != EMBEDDING_SIZE):
        raise ValueError("checkpoint model contract differs from this code")
    model = IdentityEncoder(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
