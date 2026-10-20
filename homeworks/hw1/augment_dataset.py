"""Combine Copybook ZIPs and optionally add mild image augmentations.

Example:
    python homeworks/hw1/augment_dataset.py old.zip new.zip \
        --output combined_augmented.zip --copies 3 --seed 541
"""

import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def augment(image, rng):
    """Apply a small affine change while keeping a white background."""
    angle = math.radians(rng.uniform(-6.0, 6.0))
    scale = rng.uniform(0.95, 1.05)
    shear = rng.uniform(-0.04, 0.04)
    dx, dy = rng.uniform(-2.0, 2.0, size=2)
    center = (image.width - 1) / 2.0

    def move(x, y):
        return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])

    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    skew_and_scale = np.array([
        [scale, shear, 0.0],
        [0.0, scale, 0.0],
        [0.0, 0.0, 1.0],
    ])
    forward = move(center + dx, center + dy) @ rotation @ skew_and_scale @ move(-center, -center)
    inverse = np.linalg.inv(forward)
    coefficients = tuple(inverse[:2].reshape(-1))
    result = image.transform(
        image.size, Image.Transform.AFFINE, coefficients,
        resample=Image.Resampling.BICUBIC, fillcolor=255,
    )

    thickness = rng.random()
    if thickness < 0.10:
        result = result.filter(ImageFilter.MinFilter(3))  # slightly thicker ink
    elif thickness < 0.20:
        result = result.filter(ImageFilter.MaxFilter(3))  # slightly thinner ink
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Copybook export ZIPs")
    parser.add_argument("--output", required=True, type=Path, help="output Copybook ZIP")
    parser.add_argument("--copies", type=int, default=0,
                        help="augmented copies per original image (default: 0)")
    parser.add_argument("--seed", type=int, default=541)
    args = parser.parse_args()
    if args.copies < 0:
        parser.error("--copies must be nonnegative")
    if args.output.resolve() in {p.resolve() for p in args.inputs}:
        parser.error("output must differ from every input")

    rng = np.random.default_rng(args.seed)
    records = []
    seen = {}
    duplicates = 0
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for source in args.inputs:
            with zipfile.ZipFile(source) as archive:
                metadata = json.loads(archive.read("labels.json"))
                for item in metadata:
                    image = Image.open(io.BytesIO(archive.read(f"shots/{item['file']}"))).convert("L")
                    if image.size != (64, 64):
                        raise ValueError(f"{source}: {item['file']} is not 64x64")
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if digest in seen:
                        if seen[digest] != item["label"]:
                            raise ValueError("identical images have conflicting labels")
                        duplicates += 1
                        continue
                    seen[digest] = item["label"]

                    for copy_number in range(args.copies + 1):
                        variant = image if copy_number == 0 else augment(image, rng)
                        filename = f"{len(records):05d}.png"
                        buffer = io.BytesIO()
                        variant.save(buffer, format="PNG")
                        out.writestr(f"shots/{filename}", buffer.getvalue())
                        record = dict(item)
                        record["file"] = filename
                        record["source"] = source.name
                        record["augmentation"] = copy_number
                        records.append(record)
        out.writestr("labels.json", json.dumps(records, ensure_ascii=False, indent=2))
        out.writestr("README.txt", f"{len(seen)} originals; {args.copies} augmented copies each; seed {args.seed}.\n")
    print(f"Wrote {args.output}: {len(seen)} originals, {len(records)} total images, "
          f"{duplicates} duplicate originals skipped")


if __name__ == "__main__":
    main()
