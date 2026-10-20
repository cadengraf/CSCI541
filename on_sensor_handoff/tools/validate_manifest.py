"""Validate a version-1 object-identity observation manifest (JSON Lines)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


TEXT_FIELDS = (
    "dataset",
    "capture_group",
    "camera_id",
    "track_id",
    "identity_id",
    "category",
    "image_path",
)


def validate(path: Path, check_files: bool, base_dir: Path) -> tuple[int, int, int]:
    observations = 0
    identities: set[tuple[str, str, str]] = set()
    cameras_by_identity: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    track_identity: dict[tuple[str, str, str, str], str] = {}
    identity_category: dict[tuple[str, str], str] = {}
    seen_observations: set[tuple[str, str, str, str, int]] = set()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
                raise ValueError(f"line {line_number}: schema_version must be 1")
            for field in TEXT_FIELDS:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"line {line_number}: {field} must be nonempty text")
            timestamp = row.get("timestamp_us")
            if type(timestamp) is not int or timestamp < 0:
                raise ValueError(f"line {line_number}: timestamp_us must be a nonnegative integer")
            box = row.get("bbox_xywh")
            if not isinstance(box, list) or len(box) != 4 or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in box
            ):
                raise ValueError(f"line {line_number}: bbox_xywh must contain four finite numbers")
            if box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"line {line_number}: bbox_xywh needs nonnegative x/y and positive width/height")

            track_key = (
                row["dataset"], row["capture_group"], row["camera_id"], row["track_id"]
            )
            prior_identity = track_identity.setdefault(track_key, row["identity_id"])
            if prior_identity != row["identity_id"]:
                raise ValueError(f"line {line_number}: camera-local track changes identity")
            identity_key = (row["dataset"], row["identity_id"])
            prior_category = identity_category.setdefault(identity_key, row["category"])
            if prior_category != row["category"]:
                raise ValueError(f"line {line_number}: identity changes category")
            observation_key = (*track_key, timestamp)
            if observation_key in seen_observations:
                raise ValueError(f"line {line_number}: duplicate track observation at timestamp")
            seen_observations.add(observation_key)

            image_path = Path(row["image_path"])
            if image_path.is_absolute() or ".." in image_path.parts:
                raise ValueError(f"line {line_number}: image_path must be a portable relative path")
            if check_files and not (base_dir / image_path).is_file():
                raise ValueError(f"line {line_number}: image does not exist: {image_path}")

            instance_key = (row["dataset"], row["capture_group"], row["identity_id"])
            identities.add(instance_key)
            cameras_by_identity[instance_key].add(row["camera_id"])
            observations += 1

    if observations == 0:
        raise ValueError("manifest has no observations")
    paired = sum(len(cameras) >= 2 for cameras in cameras_by_identity.values())
    return observations, len(identities), paired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON Lines observation manifest")
    parser.add_argument(
        "--check-files", action="store_true", help="check that referenced images exist"
    )
    parser.add_argument(
        "--base-dir", type=Path, default=Path.cwd(), help="base for image_path entries"
    )
    args = parser.parse_args()
    try:
        observations, identities, paired = validate(
            args.manifest, args.check_files, args.base_dir
        )
    except (OSError, ValueError) as error:
        print(f"manifest invalid: {error}", file=sys.stderr)
        return 1
    print(
        f"valid: {observations} observations, {identities} session identities, "
        f"{paired} observed by multiple cameras"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
