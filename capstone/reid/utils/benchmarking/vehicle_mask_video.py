"""Write one MP4 showing vehicle crops, predicted masks, and available truth."""

import csv
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

from train import CONFIG, path_from_config
from utils.core.datasets import ManifestItems
from .mask_eval import _image_and_mask
from .models import _load_joint

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _panel(raw_path, tensor, predicted, truth, title):
    with Image.open(raw_path) as source:
        raw = source.convert("RGB")
        raw.thumbnail((224, 224), Image.Resampling.BILINEAR)
    raw_panel = Image.new("RGB", (224, 224), (20, 20, 20))
    raw_panel.paste(raw, ((224 - raw.width) // 2, (224 - raw.height) // 2))
    original = tensor.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    original = (original.clip(0, 1) * 255).astype(np.uint8)
    overlay = original.copy()
    overlay[predicted] = (0.45 * overlay[predicted] +
                          0.55 * np.array([10, 240, 65])).astype(np.uint8)
    if truth is None:
        reference = np.full_like(original, 35)
        caption = "No VRU pixel annotation"
    else:
        reference = original.copy()
        reference[truth] = (0.45 * reference[truth] +
                            0.55 * np.array([50, 150, 255])).astype(np.uint8)
        union = np.logical_or(predicted, truth).sum()
        score = np.logical_and(predicted, truth).sum() / union if union else 0.0
        caption = f"Ground truth in blue; IoU {score:.1%}"
    canvas = Image.new("RGB", (896, 270), (20, 20, 20))
    for index, image in enumerate((raw_panel, Image.fromarray(original),
                                   Image.fromarray(overlay), Image.fromarray(reference))):
        canvas.paste(image, (224 * index, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:85], fill="white")
    draw.text((8, 256), "Raw crop             Model input             Predicted green             " + caption,
              fill="white")
    return np.asarray(canvas, dtype=np.uint8)


def main():
    settings = yaml.safe_load(CONFIG.read_text())["vru"]
    requested = settings["mask_video_checkpoint"]
    if requested == "auto":
        requested = settings["checkpoints"]["Joint aerial validated"]
        for group, label in (("matched_checkpoints", "Vehicle joint binary mask"),
                             ("warmstart_checkpoints", "Vehicle warm-start joint")):
            candidate = path_from_config(settings[group][label])
            if candidate.is_file():
                saved = torch.load(candidate, map_location="cpu", weights_only=True)
                if saved.get("training", {}).get("finished"):
                    requested = str(candidate)
    checkpoint = path_from_config(requested)
    output = path_from_config(settings["mask_video_output"])
    model = _load_joint(checkpoint).eval()
    if not model.segmentation_trained:
        raise ValueError("Mask video requires a checkpoint with a trained mask head")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    samples = []
    validation = path_from_config(settings["mask_eval_manifest"])
    with validation.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("category") in ("Small_Vehicle", "Large_Vehicle"):
                tensor, truth = _image_and_mask(validation.parent / row["image"],
                                                validation.parent / row["mask"],
                                                full_resize=model.preprocessing == "vehicle_full_resize")
                samples.append((validation.parent / row["image"], tensor, truth.numpy(),
                                f"{checkpoint.stem} / iSAID validation: {row['category']}"))
    vru = ManifestItems(path_from_config(settings["query_manifest"]),
                        preprocessing="vehicle" if model.preprocessing == "vehicle_full_resize"
                        else "imagenet")
    count = settings["mask_video_examples"]
    for index in np.linspace(0, len(vru) - 1, count, dtype=int):
        tensor, identity = vru[int(index)]
        samples.append((vru.samples[int(index)][0], tensor, None,
                        f"{checkpoint.stem} / VRU test vehicle ID {identity}"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "896x270", "-r", "4", "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with torch.inference_mode():
            for raw_path, tensor, truth, label in samples:
                logits, _ = model(tensor[None].to(device))
                foreground = 1 - logits.softmax(1)[:, :1]
                foreground = F.interpolate(foreground, size=(224, 224),
                                           mode="bilinear", align_corners=False)
                predicted = (foreground[0, 0] >= 0.5).cpu().numpy()
                frame = _panel(raw_path, tensor, predicted, truth, label)
                for _ in range(6):
                    process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed to write vehicle mask video")
        temporary.replace(output)
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    print(f"Saved {output}: {len(samples)} crops; first frames have iSAID truth, "
          "remaining VRU frames are qualitative only")
    print(f"Mask checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
