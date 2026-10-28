"""Prepare only VRAI's official identity-labeled training partition."""

import csv
import pickle
import random
import re
import tarfile
from collections import Counter
from pathlib import Path

import yaml

from train import CONFIG, path_from_config


NAME = re.compile(r"^(\d+)_(\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def prepare(root: Path, output: Path):
    annotation = root / "train_annotation.pkl"
    if not annotation.is_file():
        raise FileNotFoundError(f"Missing official {annotation}")
    images = root / "images_train"
    if not images.is_dir():
        archive = root / "images_train.tar"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing {images} and {archive}")
        print(f"Extracting {archive} into {root}", flush=True)
        with tarfile.open(archive) as tar:
            if any(not (member.name == "images_train" or member.name.startswith("images_train/"))
                   for member in tar.getmembers()):
                raise ValueError("Unexpected path in VRAI train archive")
            tar.extractall(root, filter="data")
    # Pickle is used by the dataset authors. Only open the official archive you trust.
    with annotation.open("rb") as file:
        data = pickle.load(file, encoding="latin1")
    names = data.get("train_im_names")
    if names is None or len(names) == 0:
        raise ValueError("VRAI annotation needs a nonempty train_im_names list")
    names = [Path(item.decode() if isinstance(item, bytes) else str(item)).name for item in names]
    if len(names) != len(set(names)):
        raise ValueError("VRAI train_im_names contains duplicate filenames")
    wanted = set(names)
    missing = {name for name in wanted if not (images / name).is_file()}
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} official train images; example: {next(iter(missing))}")
    rows = []
    for name in names:
        match = NAME.fullmatch(Path(name).name)
        if not match:
            raise ValueError(f"Unexpected VRAI training filename: {name}")
        identity, camera, _ = match.groups()
        rows.append((str((images / name).resolve()), identity, camera))
    counts = Counter(identity for _, identity, _ in rows)
    if any(count < 2 for count in counts.values()):
        raise ValueError("VRAI training partition contains an ID with fewer than two images")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.csv")
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("path", "id", "camera"))
        writer.writerows(rows)
    temporary.replace(output)
    print(f"Prepared {len(rows)} official VRAI training crops / {len(counts)} identities: {output}")
    print("Official test images are excluded. VRAI's public test metadata does not expose identity labels for local Rank-1/mAP.")


def prepare_disjoint(source: Path, destination: Path, seed: int = 42):
    """Split official training IDs into new train/calibration/final test IDs."""
    with source.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["path", "id", "camera"]:
            raise ValueError(f"Unexpected VRAI manifest columns: {reader.fieldnames}")
        rows = list(reader)
    by_id = {}
    for row in rows:
        by_id.setdefault(row["id"], {}).setdefault(row["camera"], []).append(row)
    eligible = [identity for identity, cameras in by_id.items()
                if "0001" in cameras and "0002" in cameras]
    if len(eligible) != len(by_id):
        raise ValueError(f"{len(by_id) - len(eligible)} VRAI IDs lack both camera views")
    identities = sorted(eligible)
    random.Random(seed).shuffle(identities)
    n_final = round(len(identities) * 0.10)
    n_calibration = round(len(identities) * 0.10)
    final_ids = set(identities[:n_final])
    calibration_ids = set(identities[n_final:n_final + n_calibration])
    train_ids = set(identities[n_final + n_calibration:])
    assert not (train_ids & calibration_ids or train_ids & final_ids or calibration_ids & final_ids)

    def selected(identity, camera, limit):
        candidates = sorted(by_id[identity][camera], key=lambda row: row["path"])
        if len(candidates) <= limit:
            return candidates
        # Spread samples across the recording so one near-duplicate burst does
        # not dominate each ID's query/gallery evaluation.
        indices = [round(i * (len(candidates) - 1) / (limit - 1)) for i in range(limit)]
        return [candidates[index] for index in indices]

    manifests = {"train_disjoint": [row for row in rows if row["id"] in train_ids]}
    for prefix, chosen_ids in (("calibration_", calibration_ids), ("", final_ids)):
        manifests[prefix + "query"] = [row for identity in sorted(chosen_ids)
                                          for row in selected(identity, "0001", 2)]
        manifests[prefix + "gallery"] = [row for identity in sorted(chosen_ids)
                                            for row in selected(identity, "0002", 4)]
    seen = set()
    for name, selected_rows in manifests.items():
        paths = {row["path"] for row in selected_rows}
        if len(paths) != len(selected_rows) or seen & paths:
            raise ValueError(f"Duplicate or shared images in VRAI {name}")
        seen.update(paths)
    destination.mkdir(parents=True, exist_ok=True)
    for name, selected_rows in manifests.items():
        output = destination / f"{name}.csv"
        temporary = output.with_suffix(".partial.csv")
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "id", "camera"])
            writer.writeheader()
            writer.writerows(selected_rows)
        temporary.replace(output)
    print(f"VRAI custom split (seed {seed}): {len(train_ids)} train IDs / "
          f"{len(calibration_ids)} calibration IDs / {len(final_ids)} final test IDs")
    print("Queries use camera 0001; galleries use camera 0002. "
          "This is an identity-disjoint split of official *train* data, not VRAI's official test benchmark.")


def main():
    settings = yaml.safe_load(CONFIG.read_text())["vrai"]
    output = path_from_config(settings["train_manifest"])
    prepare(path_from_config(settings["root"]), output)
    prepare_disjoint(output, output.parent, settings["split_seed"])


if __name__ == "__main__":
    main()
