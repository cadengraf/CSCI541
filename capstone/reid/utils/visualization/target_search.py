"""Interactive one-query ReID diagnostic using VisDrone boxes as proposals."""

import shutil
import subprocess

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageTk

from utils.core.model import JointNet
from utils.core.tracking import TemporalSelector
from utils.visualization.video import PREPROCESS, _annotations, _font


SIDEBAR_WIDTH = 286
BANNER_HEIGHT = 62


def _boxes(rows, size, minimum):
    width, height = size
    boxes = []
    for identity, x, y, box_width, box_height, category in rows:
        box = (max(0, x), max(0, y), min(width, x + box_width), min(height, y + box_height))
        if box[2] - box[0] >= minimum and box[3] - box[1] >= minimum:
            boxes.append((identity, category, box))
    return boxes


def _select_with_click(frame, boxes):
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
    except Exception as error:
        raise RuntimeError("Cannot open the selection window. Set select.target_id in config.yaml to an ID visible in the first frame.") from error
    root.title("Select one object for ReID")
    scale = min(1150 / frame.width, 700 / frame.height, 1)
    preview = frame.resize((round(frame.width * scale), round(frame.height * scale)))
    preview_draw = ImageDraw.Draw(preview)
    for _, _, (left, top, right, bottom) in boxes:
        preview_draw.rectangle((round(left * scale), round(top * scale), round(right * scale), round(bottom * scale)),
                               outline="#55dcff", width=2)
    photo = ImageTk.PhotoImage(preview)
    label = tk.Label(root, text="Click the object you want to find later. Blue boxes are VisDrone proposals.",
                     font=("DejaVu Sans", 13), padx=12, pady=8)
    label.pack()
    canvas = tk.Canvas(root, width=preview.width, height=preview.height, highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=photo)
    chosen = {"id": None}

    def click(event):
        x, y = event.x / scale, event.y / scale
        hits = [box for box in boxes if box[2][0] <= x <= box[2][2] and box[2][1] <= y <= box[2][3]]
        if not hits:
            messagebox.showinfo("No object selected", "Click inside one of the blue boxes.", parent=root)
            return
        identity, _, _ = min(hits, key=lambda item: (item[2][2] - item[2][0]) * (item[2][3] - item[2][1]))
        chosen["id"] = identity
        root.destroy()

    canvas.bind("<Button-1>", click)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.mainloop()
    if chosen["id"] is None:
        raise SystemExit("Selection cancelled")
    return chosen["id"]


def _line(draw, xy, value, font, fill="white"):
    draw.text(xy, value, font=font, fill=fill)


def _overlay_mask(frame, box, foreground):
    left, top, right, bottom = box
    mask = F.interpolate(foreground[None, None], size=(bottom - top, right - left),
                         mode="bilinear", align_corners=False)[0, 0]
    alpha = (mask.clamp(0, 1) * 145).byte().numpy()
    layer = Image.new("RGBA", (right - left, bottom - top), (0, 200, 255, 0))
    layer.putalpha(Image.fromarray(alpha, "L"))
    frame.paste(layer, (left, top), layer)


def run(config, val_root):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Install ffmpeg to create the MP4")
    if any(config[key] < 1 for key in ("frames", "stride", "fps", "min_box")):
        raise ValueError("frames, stride, fps, and min_box must be positive")
    sequence_root = val_root / "sequences"
    available = sorted(path for path in sequence_root.iterdir() if path.is_dir())
    if not available:
        raise FileNotFoundError(f"No videos found under {sequence_root}")
    sequence = config["source"] or available[0].name
    frames_dir = sequence_root / sequence
    annotation_file = val_root / "annotations" / f"{sequence}.txt"
    if not frames_dir.is_dir() or not annotation_file.is_file():
        raise FileNotFoundError(f"Missing VisDrone video or annotations for {sequence}")
    annotations = _annotations(annotation_file)
    paths = [path for path in sorted(frames_dir.glob("*.jpg")) if int(path.stem) in annotations]
    paths = paths[::config["stride"]][:config["frames"]]
    if len(paths) < 2:
        raise ValueError("The selected video needs at least two annotated sampled frames")
    with Image.open(paths[0]) as original:
        first = original.convert("RGB")
    boxes = _boxes(annotations[int(paths[0].stem)], first.size, config["min_box"])
    if not boxes:
        raise ValueError(f"No selectable objects in {paths[0]}")
    target_id = config["target_id"] if config["target_id"] is not None else _select_with_click(first, boxes)
    selected = next((item for item in boxes if item[0] == target_id), None)
    if selected is None:
        raise ValueError(f"Target ID {target_id} is not in the first sampled frame; available: {[item[0] for item in boxes]}")
    _, _, query_box = selected
    query_crop = first.crop(query_box)

    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    if not checkpoint.get("use_mask_pool", True):
        raise ValueError("Select a joint checkpoint trained with mask-guided pooling")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointNet(pretrained=False, use_mask_pool=True).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        _, query_embedding = model(PREPROCESS(query_crop).unsqueeze(0).to(device))
        query_embedding = F.normalize(query_embedding.float(), dim=1)[0]
    selector = TemporalSelector(query_box, query_embedding, int(paths[0].stem))

    output = config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    width, height = first.size
    full_width, full_height = width + SIDEBAR_WIDTH, height + BANNER_HEIGHT
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{full_width}x{full_height}", "-r", str(config["fps"]), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    font = _font(18)
    heading = _font(22)
    thumb_scale = min((SIDEBAR_WIDTH - 32) / query_crop.width, 210 / query_crop.height)
    thumbnail = query_crop.resize((max(1, round(query_crop.width * thumb_scale)),
                                   max(1, round(query_crop.height * thumb_scale))))
    searched = correct = 0
    try:
        with torch.inference_mode():
            for index, path in enumerate(paths):
                with Image.open(path) as image:
                    frame = image.convert("RGB")
                candidates = _boxes(annotations[int(path.stem)], frame.size, config["min_box"])
                if not candidates:
                    continue
                inputs = torch.stack([PREPROCESS(frame.crop(box)) for _, _, box in candidates]).to(device)
                logits, vectors = model(inputs)
                vectors = F.normalize(vectors.float(), dim=1)
                scores = vectors @ query_embedding
                frame_number = int(path.stem)
                best = (next(i for i, item in enumerate(candidates) if item[0] == target_id)
                        if index == 0 else selector.choose(candidates, vectors, scores, frame_number))
                best_id, _, best_box = candidates[best] if best is not None else (None, None, None)
                target_present = next((item for item in candidates if item[0] == target_id), None)
                rank = None
                if target_present is not None:
                    target_index = next(i for i, item in enumerate(candidates) if item[0] == target_id)
                    rank = 1 + int((scores > scores[target_index]).sum())
                if index > 0 and target_present is not None:
                    searched += 1
                    correct += best_id == target_id
                if target_present is None:
                    result = "TARGET ABSENT"
                    color = "#f4ba67"
                elif best is None:
                    result = "TRACK LOST"
                    color = "#f4ba67"
                elif best_id == target_id:
                    result = "CORRECT MATCH"
                    color = "#52e490"
                else:
                    result = "WRONG MATCH"
                    color = "#ff897a"

                if best is not None:
                    foreground = 1 - logits.softmax(dim=1)[best, 0].cpu()
                    _overlay_mask(frame, best_box, foreground)
                draw = ImageDraw.Draw(frame)
                if target_present is not None and best_id != target_id:
                    draw.rectangle(target_present[2], outline="#6fc8ff", width=4)
                    _line(draw, (target_present[2][0] + 3, max(0, target_present[2][1] - 23)),
                          "Actual target", font, "#6fc8ff")
                if best_box is not None:
                    draw.rectangle(best_box, outline=color, width=5)
                    _line(draw, (best_box[0] + 3, max(0, best_box[1] - 23)),
                          f"TRACK  {float(scores[best]):.2f}", font, color)

                canvas = Image.new("RGB", (full_width, full_height), "#13202b")
                canvas.paste(frame, (0, BANNER_HEIGHT))
                canvas.paste(thumbnail, (width + (SIDEBAR_WIDTH - thumbnail.width) // 2, BANNER_HEIGHT + 48))
                panel = ImageDraw.Draw(canvas)
                _line(panel, (14, 6), "Selected-object ReID | ShuffleNetV2 x1.5", heading)
                _line(panel, (14, 36), "Boxes: VisDrone labels   Cyan: model mask   Match: ReID + motion", font, "#c5d4dd")
                sidebar_x = width + 16
                _line(panel, (sidebar_x, BANNER_HEIGHT + 10), "SELECTED TARGET", font, "#8fdfff")
                y = BANNER_HEIGHT + 275
                _line(panel, (sidebar_x, y), result, heading, color)
                _line(panel, (sidebar_x, y + 38),
                      f"Similarity: {float(scores[best]):.3f}" if best is not None else "Similarity: --", font)
                _line(panel, (sidebar_x, y + 68), f"Target rank: {rank if rank else 'absent'}", font)
                _line(panel, (sidebar_x, y + 98), f"Candidates: {len(candidates)}", font)
                _line(panel, (sidebar_x, y + 128), f"Frame: {path.stem}", font)
                if searched:
                    _line(panel, (sidebar_x, y + 174), f"Tracked: {correct}/{searched}", font, "#c5d4dd")
                _line(panel, (sidebar_x, full_height - 74), "Green: correct", font, "#52e490")
                _line(panel, (sidebar_x, full_height - 49), "Red: wrong", font, "#ff897a")
                _line(panel, (sidebar_x, full_height - 24), "Blue: actual target", font, "#6fc8ff")
                process.stdin.write(canvas.tobytes())
                print(f"Frame {index + 1}/{len(paths)}", end="\r", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the video")
        temporary.replace(output)
        print(f"\nSaved {output}; target visible in {searched} later sampled frames, tracked correctly in {correct}.")
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
