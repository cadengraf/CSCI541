"""Evaluate foreground masks on labeled aerial object crops."""

import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.models import ShuffleNet_V2_X1_5_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from train import CONFIG, path_from_config
from utils.benchmarking.models import _load_joint
from utils.core.identity_data import NORMALIZE


def _image_and_mask(image_path: Path, mask_path: Path):
    settings = ShuffleNet_V2_X1_5_Weights.DEFAULT.transforms()
    with Image.open(image_path) as source, Image.open(mask_path) as annotation:
        image = source.convert("RGB")
        mask = annotation.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Mask size {mask.size} differs from image size {image.size}: {mask_path}")
        image = TF.resize(image, settings.resize_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, settings.resize_size, interpolation=InterpolationMode.NEAREST)
        image = TF.center_crop(image, settings.crop_size)
        mask = TF.center_crop(mask, settings.crop_size)
        return NORMALIZE(TF.to_tensor(image)), TF.pil_to_tensor(mask)[0] > 0


def main():
    settings = yaml.safe_load(CONFIG.read_text())["aerial_mask_eval"]
    if not 0 <= settings["threshold"] <= 1:
        raise ValueError("aerial_mask_eval.threshold must be between 0 and 1")
    manifest = path_from_config(settings["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing {manifest}. Make a CSV with image,mask columns; "
            "each PNG mask must have white object pixels and black background pixels."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with manifest.open(newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"image", "mask"}.issubset(reader.fieldnames):
            raise ValueError(f"{manifest} must have image,mask columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Need at least one annotated object")
    for label, checkpoint_name in settings["checkpoints"].items():
        checkpoint = path_from_config(checkpoint_name)
        if not checkpoint.is_file():
            print(f"Skipping {label}: missing {checkpoint}")
            continue
        model = _load_joint(checkpoint).eval()
        if not model.segmentation_trained:
            raise ValueError(f"Checkpoint has no trained mask head: {checkpoint}")
        model.to(device)
        counts = defaultdict(lambda: [0, 0, 0, 0, 0])
        for row in rows:
            image_path = (manifest.parent / row["image"]).resolve()
            mask_path = (manifest.parent / row["mask"]).resolve()
            image, truth = _image_and_mask(image_path, mask_path)
            with torch.inference_mode():
                logits, _ = model(image[None].to(device))
                foreground = 1 - logits.softmax(1)[:, :1]
                foreground = F.interpolate(foreground, size=truth.shape, mode="bilinear", align_corners=False)
                prediction = (foreground[0, 0] >= settings["threshold"]).cpu()
            intersection = (prediction & truth).sum().item()
            union = (prediction | truth).sum().item()
            for category in ("all", row.get("category", "unknown")):
                totals = counts[category]
                totals[0] += intersection
                totals[1] += union
                totals[2] += prediction.sum().item()
                totals[3] += truth.sum().item()
                totals[4] += 1
        for category, (intersection, union, prediction_pixels, truth_pixels, count) in counts.items():
            if not truth_pixels:
                raise ValueError("Need annotated objects with foreground pixels")
            iou = intersection / union if union else 0
            dice = 2 * intersection / (prediction_pixels + truth_pixels)
            print(f"{label} / {category}: {count} crops, IoU {iou:.2%}, Dice {dice:.2%}")
        print(f"Checkpoint: {checkpoint}; threshold: {settings['threshold']}; device: {device}")


if __name__ == "__main__":
    main()
