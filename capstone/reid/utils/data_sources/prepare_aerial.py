"""Create identity manifests from aerial datasets; does not download or copy photos."""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def images_under(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Missing image folder: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def identity_folders(root):
    rows = []
    for path in images_under(root):
        relative = path.relative_to(root)
        if len(relative.parts) >= 2:
            rows.append((path, relative.parts[0], None))
    if not rows:
        raise ValueError(f"No identity folders with images under {root}")
    return rows


def write_manifests(output_dir, train, query, gallery):
    counts = Counter(label for _, label, _ in train)
    train = [row for row in train if counts[row[1]] >= 2]
    if not train or not query or not gallery:
        raise ValueError("Need training pairs and nonempty query/gallery manifests")
    if not set(label for _, label, _ in query) & set(label for _, label, _ in gallery):
        raise ValueError("Query and gallery have no matching identities")
    keys = [{(path.resolve(), box) for path, _, box in rows} for rows in (train, query, gallery)]
    if keys[0] & (keys[1] | keys[2]) or keys[1] & keys[2]:
        raise ValueError("Train, query, and gallery contain a repeated image crop")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("query", query), ("gallery", gallery)):
        destination = output_dir / f"{name}.csv"
        with destination.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("path", "id", "x", "y", "w", "h"))
            for path, label, box in rows:
                relative = os.path.relpath(path.resolve(), destination.parent.resolve())
                writer.writerow((relative, label, *(box if box is not None else ("",) * 4)))
        print(f"{destination}: {len(rows)} images, {len(set(label for _, label, _ in rows))} IDs")


def prepare_folders(args):
    root = args.root
    train = identity_folders(root / args.train_drone) + identity_folders(root / args.train_satellite)
    query = identity_folders(root / args.query)
    gallery = identity_folders(root / args.gallery)
    write_manifests(args.output_dir, train, query, gallery)


def choose_evenly(rows, limit):
    if not limit or len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(index * (len(rows) - 1) / (limit - 1))] for index in range(limit)]


def prepare_sues(args):
    drone_root = args.root / "drone_view_512"
    satellite_root = args.root / "satellite-view"
    if not drone_root.is_dir() or not satellite_root.is_dir():
        raise FileNotFoundError("SUES-200 needs drone_view_512/ and satellite-view/")
    scenes = sorted(set(p.name for p in drone_root.iterdir() if p.is_dir()) &
                    set(p.name for p in satellite_root.iterdir() if p.is_dir()))
    valid_scenes = {}
    for scene in scenes:
        drone = images_under(drone_root / scene)
        if args.heights:
            drone = [p for p in drone if p.relative_to(drone_root / scene).parts[0] in args.heights]
        satellite = images_under(satellite_root / scene)
        if drone and satellite:
            valid_scenes[scene] = (drone, satellite)
    if len(valid_scenes) <= args.test_scenes:
        raise ValueError(f"Need more than {args.test_scenes} complete scenes; found {len(valid_scenes)}")
    held_out = set(random.Random(args.seed).sample(sorted(valid_scenes), args.test_scenes))
    train, query, gallery = [], [], []
    for scene, (drone, satellite) in valid_scenes.items():
        if scene in held_out:
            query.extend((p, scene, None) for p in choose_evenly(drone, args.max_query_per_scene))
            gallery.append((satellite[0], scene, None))
        else:
            train.extend((p, scene, None) for p in choose_evenly(drone, args.max_train_per_scene))
            train.append((satellite[0], scene, None))
    write_manifests(args.output_dir, train, query, gallery)
    print("SUES-200: seeded scene-level holdout for this project; these are not the official benchmark splits")


def visdrone_tracks(root, frame_step, min_box):
    annotations = root / "annotations"
    sequences = root / "sequences"
    if not annotations.is_dir() or not sequences.is_dir():
        raise FileNotFoundError(f"{root} needs annotations/ and sequences/ from VisDrone-MOT")
    tracks = defaultdict(list)
    for annotation in sorted(annotations.glob("*.txt")):
        frames = images_under(sequences / annotation.stem)
        by_number = {int(p.stem): p for p in frames if p.stem.isdigit()}
        with annotation.open(newline="") as file:
            for fields in csv.reader(file):
                if len(fields) < 8:
                    continue
                frame, target, x, y, width, height, score, category = (int(float(v)) for v in fields[:8])
                if target <= 0 or score != 1 or category < 1 or category > 10:
                    continue
                if width < min_box or height < min_box or (frame - 1) % frame_step:
                    continue
                path = by_number.get(frame) or (frames[frame - 1] if 0 < frame <= len(frames) else None)
                if path is None:
                    raise FileNotFoundError(f"Missing frame {frame} for {annotation.stem}")
                tracks[f"{annotation.stem}:{target}"].append((frame, path, (x, y, width, height)))
    return tracks


def prepare_visdrone(args):
    if args.train_root.resolve() == args.val_root.resolve():
        raise ValueError("VisDrone train and validation roots must differ")
    training = visdrone_tracks(args.train_root, args.frame_step, args.min_box)
    validation = visdrone_tracks(args.val_root, args.frame_step, args.min_box)
    if set(training) & set(validation):
        raise ValueError("VisDrone train and validation reuse sequence/track IDs")
    train, query, gallery = [], [], []
    for label, observations in training.items():
        observations = sorted(observations)
        if len(observations) >= 2:
            train.extend((path, label, box) for _, path, box in choose_evenly(observations, args.max_train_per_track))
    for label, observations in validation.items():
        observations = sorted(observations)
        if len(observations) < 2 or observations[-1][0] - observations[0][0] < args.min_eval_gap:
            continue
        _, first_path, first_box = observations[0]
        _, last_path, last_box = observations[-1]
        query.append((first_path, label, first_box))
        gallery.append((last_path, label, last_box))
    write_manifests(args.output_dir, train, query, gallery)
    print("VisDrone evaluation uses separated frames of the same tracks, within validation videos")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    folders = sub.add_parser("folders", help="University-1652 or DenseUAV identity folders")
    folders.add_argument("--root", type=Path, required=True)
    folders.add_argument("--output-dir", type=Path, required=True)
    folders.add_argument("--train-drone", type=Path, default=Path("train/drone"))
    folders.add_argument("--train-satellite", type=Path, default=Path("train/satellite"))
    folders.add_argument("--query", type=Path, default=Path("test/query_drone"))
    folders.add_argument("--gallery", type=Path, default=Path("test/gallery_satellite"))

    sues = sub.add_parser("sues", help="SUES-200 scene-level drone/satellite split")
    sues.add_argument("--root", type=Path, required=True)
    sues.add_argument("--output-dir", type=Path, required=True)
    sues.add_argument("--test-scenes", type=int, default=40)
    sues.add_argument("--seed", type=int, default=42)
    sues.add_argument("--heights", nargs="*", choices=("150", "200", "250", "300"))
    sues.add_argument("--max-train-per-scene", type=int, default=24)
    sues.add_argument("--max-query-per-scene", type=int, default=8)

    visdrone = sub.add_parser("visdrone", help="VisDrone-MOT tracked-object crops")
    visdrone.add_argument("--train-root", type=Path, required=True)
    visdrone.add_argument("--val-root", type=Path, required=True)
    visdrone.add_argument("--output-dir", type=Path, required=True)
    visdrone.add_argument("--frame-step", type=int, default=10)
    visdrone.add_argument("--min-box", type=int, default=16)
    visdrone.add_argument("--max-train-per-track", type=int, default=12)
    visdrone.add_argument("--min-eval-gap", type=int, default=10)

    args = parser.parse_args(argv)
    if args.dataset == "folders":
        prepare_folders(args)
    elif args.dataset == "sues":
        if min(args.test_scenes, args.max_train_per_scene, args.max_query_per_scene) < 1:
            parser.error("scene counts and per-scene limits must be positive")
        prepare_sues(args)
    else:
        if min(args.frame_step, args.min_box, args.max_train_per_track, args.min_eval_gap) < 1:
            parser.error("VisDrone sampling limits must be positive")
        if args.max_train_per_track < 2:
            parser.error("--max-train-per-track must be at least 2")
        prepare_visdrone(args)


if __name__ == "__main__":
    main()
