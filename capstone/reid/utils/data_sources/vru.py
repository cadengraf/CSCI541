"""Prepare VRU's official identity-disjoint training list."""

import csv
from collections import Counter
from pathlib import Path

import yaml

from train import CONFIG, path_from_config


def prepare(root: Path, output: Path):
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
    counts = Counter(row[1] for row in train_rows)
    if any(count < 2 for count in counts.values()):
        raise ValueError("VRU has a training identity with fewer than two images")
    rows = []
    for image_id, identity in train_rows:
        path = root / "Pic" / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing VRU image {path}")
        rows.append((str(path.resolve()), identity))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.csv")
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("path", "id"))
        writer.writerows(rows)
    temporary.replace(output)
    print(f"Prepared {len(rows)} official VRU training crops / {len(train_ids)} identities: {output}")
    print(f"Excluded {len(test_rows)} official test crops / {len(test_ids)} identities")


def main():
    settings = yaml.safe_load(CONFIG.read_text())["vru"]
    prepare(path_from_config(settings["root"]), path_from_config(settings["train_manifest"]))


if __name__ == "__main__":
    main()
