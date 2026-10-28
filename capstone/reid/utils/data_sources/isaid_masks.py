"""Prepare a small iSAID validation mask diagnostic from matching DOTA images."""

import csv
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw

from train import path_from_config


CATEGORIES = ("Small_Vehicle", "Large_Vehicle", "plane", "ship")
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def prepare(annotation: Path, output: Path, scenes: int = 40, per_category: int = 4):
    data = json.loads(annotation.read_text())
    images = data["images"][:scenes]
    response = requests.get(ROWS_URL, params={"dataset": "finedet/dota", "config": "v1.0",
                                              "split": "validation", "offset": 0,
                                              "length": scenes}, timeout=90)
    response.raise_for_status()
    rows = response.json()["rows"]
    if len(rows) != scenes:
        raise ValueError(f"Expected {scenes} mirror rows; got {len(rows)}")
    for index, (scene, row) in enumerate(zip(images, rows)):
        mirror = row["row"]["image"]
        if row["row_idx"] != index or (mirror["width"], mirror["height"]) != (scene["width"], scene["height"]):
            raise ValueError(f"DOTA mirror/iSAID order or dimensions differ at scene {index}")

    annotations = defaultdict(list)
    for item in data["annotations"]:
        if (item["image_id"] < scenes and item["category_name"] in CATEGORIES
                and item["bbox"][2] >= 32 and item["bbox"][3] >= 32
                and item["segmentation"]):
            annotations[item["image_id"]].append(item)
    chosen = []
    for category in CATEGORIES:
        candidates = []
        for scene_id, items in annotations.items():
            matching = [item for item in items if item["category_name"] == category]
            if matching:
                best = max(matching, key=lambda item: item["area"])
                candidates.append((scene_id, best))
        # One instance per scene/category limits near-duplicate evaluation crops.
        chosen.extend(sorted(candidates, key=lambda pair: pair[0])[:per_category])
    if len(chosen) != len(CATEGORIES) * per_category:
        raise ValueError("Not enough labeled instances in the chosen validation scenes")

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_id, item in chosen:
        scene = images[scene_id]
        image_name = f"{Path(scene['file_name']).stem}.jpg"
        image_file = output / image_name
        if not image_file.is_file():
            url = rows[scene_id]["row"]["image"]["src"]
            downloaded = requests.get(url, timeout=120)
            downloaded.raise_for_status()
            with Image.open(BytesIO(downloaded.content)) as source:
                if source.size != (scene["width"], scene["height"]):
                    raise ValueError(f"Mirror dimensions changed for {image_name}")
                source.convert("RGB").save(image_file, quality=95)
        with Image.open(image_file) as source:
            x, y, width, height = item["bbox"]
            margin = round(max(width, height) * 0.25)
            box = (max(0, int(x) - margin), max(0, int(y) - margin),
                   min(source.width, int(x + width) + margin),
                   min(source.height, int(y + height) + margin))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Invalid crop box for annotation {item['id']}")
            crop = source.crop(box)
            mask = Image.new("L", crop.size, 0)
            draw = ImageDraw.Draw(mask)
            for polygon in item["segmentation"]:
                points = [(polygon[i] - box[0], polygon[i + 1] - box[1])
                          for i in range(0, len(polygon) - 1, 2)]
                if len(points) >= 3:
                    draw.polygon(points, fill=255)
            if not mask.getbbox():
                raise ValueError(f"Empty mask for annotation {item['id']}")
            stem = f"{scene_id:03d}_{item['id']}"
            crop_name, mask_name = f"{stem}_image.png", f"{stem}_mask.png"
            crop.save(output / crop_name)
            mask.save(output / mask_name)
            records.append({"image": crop_name, "mask": mask_name,
                            "category": item["category_name"],
                            "scene": scene["file_name"], "annotation_id": item["id"]})
    manifest = output / "eval.csv"
    temporary = output / "eval.partial.csv"
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(manifest)
    print(f"Prepared {len(records)} iSAID validation object masks: {manifest}")
    print("Uses official iSAID polygons and matched, re-encoded DOTA mirror JPEGs; local diagnostic only.")


def main():
    prepare(path_from_config("data/aerial/isaid/val/Annotations/iSAID_val.json"),
            path_from_config("data/aerial/masks"))


if __name__ == "__main__":
    main()
