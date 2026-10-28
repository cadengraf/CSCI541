"""Render one BuckTales selected-object MP4 with masks and ReID scores."""

import shutil
import subprocess
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from utils.crossdrone.retrieval import Observations, crop, load_model
from utils.visualization.video import _font


def _tile(image, size):
    tile = Image.new("RGB", size, "#23313c")
    copy = image.copy()
    usable_width, usable_height = size[0] - 20, size[1] - 50
    scale = min(usable_width / copy.width, usable_height / copy.height)
    copy = copy.resize((max(1, round(copy.width * scale)),
                        max(1, round(copy.height * scale))), Image.Resampling.BICUBIC)
    tile.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2 + 12))
    return tile


def _masked_crop(row, foreground):
    image = crop(row)
    mask = F.interpolate(foreground[None, None], size=(image.height, image.width),
                         mode="bilinear", align_corners=False)[0, 0]
    layer = Image.new("RGBA", image.size, (0, 200, 255, 0))
    layer.putalpha(Image.fromarray((mask.clamp(0, 1) * 130).byte().numpy(), "L"))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


@torch.inference_mode()
def _vectors_and_masks(model, rows, batch_size, device):
    loader = DataLoader(Observations(rows), batch_size=batch_size)
    vectors, masks = [], []
    for images, _ in loader:
        logits, features = model(images.to(device))
        vectors.append(F.normalize(features.float(), dim=1).cpu())
        masks.append((1 - logits.softmax(dim=1)[:, 0]).cpu())
    return torch.cat(vectors), torch.cat(masks)


def render(config, rows):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write the cross-drone MP4")
    pair = config["pair"] or sorted({row.pair for row in rows})[0]
    pair_rows = [row for row in rows if row.pair == pair]
    if not pair_rows:
        raise ValueError(f"No observations for pair {pair}")
    cameras = sorted({row.camera for row in pair_rows})
    source_camera = config["source_camera"] or cameras[0]
    if source_camera not in cameras:
        raise ValueError(f"Camera {source_camera} is not in pair {pair}: {cameras}")
    destination_camera = next(camera for camera in cameras if camera != source_camera)
    source = sorted((row for row in pair_rows if row.camera == source_camera),
                    key=lambda row: (row.frame, row.identity))
    destination = [row for row in pair_rows if row.camera == destination_camera]
    known = {row.identity for row in destination}
    source = [row for row in source if row.identity in known]
    if not source:
        raise ValueError(f"{pair} has no IDs shared by {source_camera} and {destination_camera}")
    target_id = config["target_id"] or source[0].identity
    selected = next((row for row in source if row.identity == target_id), None)
    if selected is None:
        raise ValueError(f"Target {target_id} is not present in source camera {source_camera}")
    by_frame = defaultdict(list)
    for row in destination:
        by_frame[row.frame].append(row)
    frames = sorted(by_frame)[::config["video_stride"]][:config["video_frames"]]
    if not frames:
        raise ValueError("No destination frames to render")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config["checkpoint"], device)
    query_vectors, query_masks = _vectors_and_masks(
        model, [selected], config["batch_size"], device)
    query_vector = query_vectors[0]
    query_image = _masked_crop(selected, query_masks[0])
    output = config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    width, height = 1280, 720
    process = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(config["fps"]), "-i", "-", "-an",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(temporary)],
        stdin=subprocess.PIPE)
    font = _font(20)
    heading = _font(28)
    correct = visible = 0
    try:
        with torch.inference_mode():
            for number, frame in enumerate(frames, 1):
                candidates = by_frame[frame]
                vectors, masks = _vectors_and_masks(
                    model, candidates, config["batch_size"], device)
                scores = vectors @ query_vector
                order = scores.argsort(descending=True).tolist()
                top = order[0]
                target_present = any(row.identity == target_id for row in candidates)
                if target_present:
                    visible += 1
                    correct += candidates[top].identity == target_id
                canvas = Image.new("RGB", (width, height), "#13202b")
                draw = ImageDraw.Draw(canvas)
                draw.text((20, 12), f"Cross-drone ReID  |  {pair}  |  frame {frame}", font=heading, fill="white")
                draw.text((20, 56), f"Selected: {source_camera} / ID {target_id}    Search: {destination_camera}",
                          font=font, fill="#b9d5e8")
                canvas.paste(_tile(query_image, (300, 520)), (20, 110))
                draw = ImageDraw.Draw(canvas)
                draw.text((34, 590), "SELECTED OBJECT", font=font, fill="#8fdfff")
                for rank, index in enumerate(order[:3]):
                    row = candidates[index]
                    x = 350 + 305 * rank
                    canvas.paste(_tile(_masked_crop(row, masks[index]), (285, 400)), (x, 145))
                    color = "#52e490" if row.identity == target_id else "#ff897a"
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((x, 145, x + 285, 545), outline=color, width=4)
                    draw.text((x + 10, 560), f"#{rank + 1}  ID {row.identity}", font=font, fill=color)
                    draw.text((x + 10, 590), f"similarity {float(scores[index]):.3f}", font=font, fill="white")
                status = ("CORRECT" if candidates[top].identity == target_id else "WRONG") if target_present else "TARGET ABSENT"
                draw.text((20, 664), f"{status}    Top-1 when visible: {correct}/{visible}    "
                          "Cyan: predicted foreground", font=font, fill="white")
                process.stdin.write(canvas.tobytes())
                print(f"Video frame {number}/{len(frames)}", end="\r", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the cross-drone video")
        temporary.replace(output)
        print(f"\nSaved {output}; visible frames {visible}, top-1 correct {correct}", flush=True)
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
