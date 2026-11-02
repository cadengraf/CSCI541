"""Prepare resumable iSAID-train vehicle instance crops from a DOTA image mirror."""

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from PIL import Image, ImageDraw
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from train import CONFIG, path_from_config

ROWS_URL = "https://datasets-server.huggingface.co/rows"
VEHICLE_CATEGORIES = {2: "Large_Vehicle", 3: "Small_Vehicle"}


def _request_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _get(session, url, **kwargs):
    """Respect API throttling instead of exhausting urllib3's short retries."""
    host = urlparse(url).hostname or ""
    if os.environ.get("HF_TOKEN") and (host == "huggingface.co" or
                                       host.endswith(".huggingface.co")):
        kwargs["headers"] = {**kwargs.get("headers", {}),
                             "Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    for attempt in range(5):
        response = session.get(url, **kwargs)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        if attempt == 4:
            raise RuntimeError("Hugging Face is still rate limiting downloads. "
                               "Re-run `python prepare.py vehicle-masks` later; completed crops are kept.")
        retry_after = response.headers.get("Retry-After", "")
        delay = float(retry_after) if retry_after.isdigit() else min(60 * 2 ** attempt, 300)
        print(f"Hugging Face rate limit; waiting {delay:.0f}s before retry "
              f"({attempt + 1}/4). Prepared crops are safe on disk.", flush=True)
        time.sleep(delay)


def _mirror_row(session, scene_id, blocks):
    """Read up to 100 DOTA rows per API request (the documented maximum)."""
    offset = scene_id // 100 * 100
    if offset not in blocks or time.monotonic() - blocks[offset][0] > 900:
        response = _get(session, ROWS_URL, params={"dataset": "finedet/dota",
                         "config": "v1.0", "split": "train", "offset": offset,
                         "length": 100}, timeout=90)
        blocks[offset] = (time.monotonic(), {
            record["row_idx"]: record["row"] for record in response.json()["rows"]})
    try:
        return blocks[offset][1][scene_id]
    except KeyError as error:
        raise ValueError(f"DOTA mirror missing row {scene_id}") from error


def _chosen(items, limit, seed):
    large = [a for a in items if a["category_id"] == 2]
    small = [a for a in items if a["category_id"] == 3]
    order = lambda a: hashlib.sha256(f"{seed}:{a['id']}".encode()).digest()
    large.sort(key=order)
    small.sort(key=order)
    picked = large[:limit // 2] + small[:limit - min(limit // 2, len(large))]
    if len(picked) < limit:
        picked += large[limit // 2:limit // 2 + limit - len(picked)]
    return picked


def _verify_scene(row, annotations):
    """Cross-check DOTA vehicle boxes against iSAID polygons for scene alignment."""
    objects = row["objects"]
    dota = [box for box, category in zip(objects["bbox"], objects["category"])
            if category in (9, 10)]
    if not dota:
        return False
    isaid = [a["bbox"] for a in annotations]
    def near(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return (abs(ax + aw / 2 - bx - bw / 2) <= max(aw, bw) and
                abs(ay + ah / 2 - by - bh / 2) <= max(ah, bh))
    matched = sum(any(near(box, target) for target in isaid) for box in dota)
    return matched / len(dota) >= 0.8


def prepare(annotation: Path, output: Path, scenes: int, per_scene: int,
            min_side: int, seed: int):
    data = json.loads(annotation.read_text())
    if scenes < 1 or per_scene < 1 or min_side < 1:
        raise ValueError("vehicle_masks scenes, per_scene, min_side must be positive")
    grouped = defaultdict(list)
    all_vehicles = defaultdict(list)
    for item in data["annotations"]:
        if item["category_id"] in VEHICLE_CATEGORIES:
            all_vehicles[item["image_id"]].append(item)
            if (isinstance(item.get("segmentation"), list) and item["segmentation"]
                    and min(item["bbox"][2:]) >= min_side):
                grouped[item["image_id"]].append(item)
    candidates = [image for image in data["images"] if len(grouped[image["id"]]) >= per_scene]
    candidates.sort(key=lambda image: hashlib.sha256(
        f"{seed}:{image['file_name']}".encode()).digest())
    val = json.loads((annotation.parents[2] / "val" / "Annotations" / "iSAID_val.json").read_text())
    val_names = {image["file_name"] for image in val["images"]}
    session = _request_session()
    blocks = {}
    crop_folder = output / "train"
    crop_folder.mkdir(parents=True, exist_ok=True)
    records = []
    accepted = skipped = 0
    for scene in candidates:
        if accepted >= scenes:
            break
        if scene["file_name"] in val_names:
            raise ValueError(f"iSAID train/validation scene overlap: {scene['file_name']}")
        selected = _chosen(grouped[scene["id"]], per_scene, seed)
        names = [(f"{scene['id']:04d}_{a['id']}_image.jpg",
                  f"{scene['id']:04d}_{a['id']}_mask.png") for a in selected]
        if not all((crop_folder / image).is_file() and (crop_folder / mask).is_file()
                   for image, mask in names):
            mirror = _mirror_row(session, scene["id"], blocks)
            if not _verify_scene(mirror, all_vehicles[scene["id"]]):
                skipped += 1
                continue  # No shared DOTA vehicle labels to verify this scene.
            if mirror["width"] * mirror["height"] > 50_000_000:
                skipped += 1
                continue  # Avoid decoding exceptionally large source images.
            response = _get(session, mirror["image"]["src"], timeout=180)
            with Image.open(BytesIO(response.content)) as source:
                if source.size != (mirror["width"], mirror["height"]):
                    raise ValueError(f"Mirror image size mismatch: {scene['file_name']}")
                image = source.convert("RGB")
            for item, (image_name, mask_name) in zip(selected, names):
                x, y, width, height = item["bbox"]
                margin = round(max(width, height) * 0.35)
                box = (max(0, int(x) - margin), max(0, int(y) - margin),
                       min(image.width, int(x + width) + margin),
                       min(image.height, int(y + height) + margin))
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"Invalid vehicle box: {scene['file_name']} / {item['id']}")
                crop = image.crop(box)
                mask = Image.new("L", crop.size, 0)
                draw = ImageDraw.Draw(mask)
                for polygon in item["segmentation"]:
                    points = [(polygon[i] - box[0], polygon[i + 1] - box[1])
                              for i in range(0, len(polygon) - 1, 2)]
                    if len(points) >= 3:
                        draw.polygon(points, fill=255)
                if not mask.getbbox():
                    raise ValueError(f"Empty vehicle mask: {scene['file_name']} / {item['id']}")
                image_path, mask_path = crop_folder / image_name, crop_folder / mask_name
                image_partial = image_path.with_name(image_path.stem + ".partial.jpg")
                mask_partial = mask_path.with_name(mask_path.stem + ".partial.png")
                crop.save(image_partial, quality=95)
                mask.save(mask_partial)
                image_partial.replace(image_path)
                mask_partial.replace(mask_path)
        records.extend({"image": f"train/{image_name}", "mask": f"train/{mask_name}",
                        "category": VEHICLE_CATEGORIES[item["category_id"]],
                        "scene": scene["file_name"], "annotation_id": item["id"]}
                       for item, (image_name, mask_name) in zip(selected, names))
        accepted += 1
        if accepted % 10 == 0:
            print(f"Prepared {accepted}/{scenes} verified iSAID train scenes", flush=True)
    if accepted < scenes:
        raise ValueError(f"Only {accepted}/{scenes} verified vehicle scenes available")
    manifest = output / "train.csv"
    temporary = output / "train.partial.csv"
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("image", "mask", "category", "scene", "annotation_id"))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(manifest)
    print(f"Prepared {len(records)} binary vehicle masks from {accepted} iSAID train scenes: {manifest}")
    print(f"Skipped {skipped} scenes lacking a reliable DOTA/iSAID vehicle-box check or too large to decode")


def main():
    settings = yaml.safe_load(CONFIG.read_text())["vehicle_masks"]
    prepare(path_from_config(settings["annotation"]), path_from_config(settings["output_dir"]),
            settings["scenes"], settings["per_scene"], settings["min_side"], settings["seed"])


if __name__ == "__main__":
    main()
