"""Render a labeled VisDrone clip with ShuffleNet masks and ReID similarity."""

import csv
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from utils.core.model import JointNet
from utils.core.identity_data import NORMALIZE


PREPROCESS = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZE])


def _annotations(path):
    by_frame = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.reader(file):
            if len(row) < 8:
                continue
            frame, identity, x, y, width, height, valid, category = map(int, row[:8])
            if valid == 1 and identity > 0 and 1 <= category <= 10:
                by_frame[frame].append((identity, x, y, width, height, category))
    return by_frame


def _font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size) if Path(path).is_file() else ImageFont.load_default()


def render(config, val_root):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Install ffmpeg to create an MP4 video")
    sequence_root = val_root / "sequences"
    available = sorted(path for path in sequence_root.iterdir() if path.is_dir())
    if not available:
        raise FileNotFoundError(f"No VisDrone videos found under {sequence_root}")
    sequence = config["source"] or available[0].name
    frames_dir = sequence_root / sequence
    annotations_file = val_root / "annotations" / f"{sequence}.txt"
    if not frames_dir.is_dir() or not annotations_file.is_file():
        raise FileNotFoundError(f"Missing VisDrone frames or annotations for {sequence}")
    annotations = _annotations(annotations_file)
    frames = [path for path in sorted(frames_dir.glob("*.jpg")) if int(path.stem) in annotations]
    frames = frames[::config["stride"]][:config["frames"]]
    if not frames:
        raise ValueError(f"No annotated frames in {sequence}")

    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    if not checkpoint.get("use_mask_pool", True):
        raise ValueError("Video checkpoint has mask-guided pooling disabled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointNet(pretrained=False, use_mask_pool=True).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output = config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    banner_height = 58
    history = {}
    first = Image.open(frames[0])
    width, height = first.size
    first.close()
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height + banner_height}", "-r", str(config["fps"]), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    font = _font(17)
    title_font = _font(19)
    try:
        with torch.inference_mode():
            for number, path in enumerate(frames, 1):
                frame = Image.open(path).convert("RGB")
                tracks = sorted(annotations[int(path.stem)], key=lambda row: row[4] * row[5], reverse=True)
                tracks = [row for row in tracks if row[3] >= config["min_box"] and row[4] >= config["min_box"]]
                tracks = tracks[:config["max_objects"]]
                boxes, crops = [], []
                for identity, x, y, box_width, box_height, category in tracks:
                    box = (max(0, x), max(0, y), min(width, x + box_width), min(height, y + box_height))
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    boxes.append((identity, category, box))
                    crops.append(PREPROCESS(frame.crop(box)))
                if crops:
                    logits, vectors = model(torch.stack(crops).to(device))
                    probabilities = (1 - logits.softmax(dim=1)[:, 0]).cpu()
                    vectors = F.normalize(vectors.float(), dim=1).cpu()
                draw = ImageDraw.Draw(frame)
                current = {}
                for index, (identity, category, box) in enumerate(boxes):
                    left, top, right, bottom = box
                    mask = probabilities[index][None, None]
                    mask = F.interpolate(mask, size=(bottom - top, right - left), mode="bilinear", align_corners=False)[0, 0]
                    # A translucent heat map exposes weak masks instead of hiding them behind a hard threshold.
                    alpha = (mask.clamp(0, 1) * 135).byte().numpy()
                    layer = Image.new("RGBA", (right - left, bottom - top), (0, 200, 255, 0))
                    layer.putalpha(Image.fromarray(alpha, "L"))
                    frame.paste(layer, (left, top), layer)
                    current[identity] = (category, vectors[index])
                    candidates = [(old_id, float(torch.dot(vectors[index], old_vector)))
                                  for old_id, (old_category, old_vector) in history.items()
                                  if old_category == category]
                    match_id, score = max(candidates, key=lambda pair: pair[1]) if candidates else (None, 0.0)
                    color = "#53e38b" if match_id == identity else "#ffb454" if match_id else "#8bd3ff"
                    label = f"{identity}>{match_id} {score:.2f}" if match_id else f"{identity} new"
                    draw.rectangle(box, outline=color, width=3)
                    text_y = max(0, top - 22)
                    text_width = int(draw.textlength(label, font=font)) + 8
                    draw.rectangle((left, text_y, min(width, left + text_width), text_y + 22), fill="#14212c")
                    draw.text((left + 4, text_y + 1), label, font=font, fill=color)
                history = current
                canvas = Image.new("RGB", (width, height + banner_height), "#14212c")
                canvas.paste(frame, (0, banner_height))
                header = ImageDraw.Draw(canvas)
                header.text((14, 5), "ShuffleNetV2 x1.5 | mask-guided 128D ReID", font=title_font, fill="white")
                header.text((14, 31), "VisDrone boxes + IDs  |  Cyan: model mask  |  ID>closest prior ID, cosine score", font=font, fill="#c7d5dd")
                process.stdin.write(canvas.tobytes())
                print(f"Frame {number}/{len(frames)}", end="\r", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the demo")
        temporary.replace(output)
        print(f"\nSaved {output}")
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
