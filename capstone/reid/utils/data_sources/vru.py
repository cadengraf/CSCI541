"""Prepare VRU's official train IDs and a reproducible full-test retrieval split."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

import yaml

from train import CONFIG, path_from_config


def _write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.csv")
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("path", "id"))
        writer.writerows(rows)
    temporary.replace(path)


def prepare(root: Path, output: Path, query_output: Path, gallery_output: Path, seed: int):
    split = root / "train_test_split"
    train_file, test_file = split / "train.txt", split / "test.txt"
    if not train_file.is_file() or not test_file.is_file():
        raise FileNotFoundError(f"Missing official VRU train/test lists under {split}")
    train_rows = [line.split() for line in train_file.read_text().splitlines() if line.strip()]
    test_rows = [line.split() for line in test_file.read_text().splitlines() if line.strip()]
    if any(len(row) != 2 for row in train_rows + test_rows):
        raise ValueError("VRU split rows must contain image ID and physical vehicle ID")
    train_ids = {row[1] for row in train_rows}
    test_ids = {row[1] for row in test_rows}
    if train_ids & test_ids:
        raise ValueError("VRU train/test identities overlap")
    if len({row[0] for row in train_rows}) != len(train_rows):
        raise ValueError("Duplicate training image in VRU list")
    if len({row[0] for row in test_rows}) != len(test_rows):
        raise ValueError("Duplicate test image in VRU list")
    if {row[0] for row in train_rows} & {row[0] for row in test_rows}:
        raise ValueError("VRU train/test images overlap")
    counts = Counter(row[1] for row in train_rows)
    if any(count < 2 for count in counts.values()):
        raise ValueError("VRU has a training identity with fewer than two images")
    def image_rows(source):
        rows = []
        for image_id, identity in source:
            path = root / "Pic" / f"{image_id}.jpg"
            if not path.is_file():
                raise FileNotFoundError(f"Missing VRU image {path}")
            rows.append((str(path.resolve()), identity))
        return rows

    rows = image_rows(train_rows)
    test_images = image_rows(test_rows)
    by_identity = {}
    for path, identity in test_images:
        by_identity.setdefault(identity, []).append(path)
    if any(len(paths) < 2 for paths in by_identity.values()):
        raise ValueError("Every VRU test identity needs a query and gallery image")
    gallery_paths = set()
    for identity, paths in by_identity.items():
        # Paper protocol: exactly one gallery image per test identity. The
        # public list does not prescribe which image; this fixed choice does.
        gallery_paths.add(min(paths, key=lambda path: hashlib.sha256(
            f"{seed}:{identity}:{Path(path).stem}".encode()).digest()))
    query_rows = [(path, identity) for path, identity in test_images if path not in gallery_paths]
    gallery_rows = [(path, identity) for path, identity in test_images if path in gallery_paths]
    if len(gallery_rows) != len(test_ids) or len(query_rows) + len(gallery_rows) != len(test_rows):
        raise AssertionError("VRU retrieval partition is incomplete")
    _write_manifest(output, rows)
    _write_manifest(query_output, query_rows)
    _write_manifest(gallery_output, gallery_rows)
    print(f"Prepared {len(rows)} official VRU training crops / {len(train_ids)} identities: {output}")
    print(f"VRU full test: {len(query_rows)} queries, {len(gallery_rows)} gallery images, "
          f"{len(test_ids)} unseen identities (gallery seed {seed})")


def main():
    settings = yaml.safe_load(CONFIG.read_text())["vru"]
    prepare(path_from_config(settings["root"]), path_from_config(settings["train_manifest"]),
            path_from_config(settings["query_manifest"]),
            path_from_config(settings["gallery_manifest"]), int(settings["gallery_seed"]))


if __name__ == "__main__":
    main()
