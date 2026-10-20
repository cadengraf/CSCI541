"""Build a larger vehicle-only training pool from AICity development trials.

Only S01 and S03 are used. Validation S02 and test S04 remain untouched.
The source benchmark provides cross-camera pairs; crop files are generated in
the ignored data directory and are subject to the AICity dataset license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "on_sensor_handoff/data/aicity_train_v1")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--max-trials", type=int,
                        help="extract only the first N trials for a decoding smoke test")
    args = parser.parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.relative_to(base_dir)
    source_bytes = args.source_manifest.read_bytes()
    rows = [json.loads(line) for line in source_bytes.splitlines()]
    rows = [row for row in rows if row["benchmark_split"] == "development"
            and row["scenario"] in ("S01", "S03") and row["present"]]
    if not rows:
        raise ValueError("no present S01/S03 development trials")
    if args.max_trials is not None:
        if args.max_trials < 1:
            parser.error("max-trials must be positive")
        rows = rows[:args.max_trials]
    captures: dict[Path, cv2.VideoCapture] = {}
    observations: dict[str, dict] = {}
    pairs: list[dict] = []

    def add_crop(row: dict, camera: str, video_path: Path,
                 frame_number: int, box_xyxy: list[float]) -> str:
        identity = int(row["target_id"])
        key = f"{camera}/{frame_number}/{identity}"
        if key in observations:
            return key
        capture = captures.get(video_path)
        if capture is None:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError(f"cannot open {video_path}")
            captures[video_path] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"cannot read frame {frame_number} from {video_path}")
        height, width = image.shape[:2]
        x0, y0, x1, y1 = (round(value) for value in box_xyxy)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError(f"invalid box in {row['trial_id']}: {box_xyxy}")
        crop = image[y0:y1, x0:x1]
        camera_dir = output_dir / "crops" / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        image_path = camera_dir / f"f{frame_number:06d}_id{identity}.jpg"
        if not cv2.imwrite(str(image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise ValueError(f"cannot write {image_path}")
        observations[key] = {
            "schema_version": 1,
            "dataset": "AICity22_Track1_MTMC_Tracking",
            "capture_group": row["scenario"],
            "camera_id": camera,
            "track_id": str(identity),
            "identity_id": f"{row['scenario']}/{identity}",
            "category": "vehicle",
            "timestamp_us": (frame_number - 1) * 100000,
            "image_path": str(image_path.relative_to(base_dir)),
            "bbox_xywh": [0, 0, crop.shape[1], crop.shape[0]],
            "observation_id": key,
            "frame_index": frame_number,
            "split": "train",
        }
        return key

    try:
        for row in rows:
            source = row["source_samples"][-1]
            query = add_crop(row, row["source_camera"], Path(row["source_video"]),
                             int(source["frame"]), source["bbox_xyxy"])
            positive = add_crop(row, row["destination_camera"],
                                Path(row["destination_video"]),
                                int(row["destination_frame"]),
                                row["destination_bbox_xyxy"])
            if observations[query]["camera_id"] == observations[positive]["camera_id"]:
                raise ValueError(f"same-camera pair: {row['trial_id']}")
            pairs.append({"trial_id": row["trial_id"], "query": query,
                          "positive": positive,
                          "identity_id": observations[query]["identity_id"]})
    finally:
        for capture in captures.values():
            capture.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("observations.jsonl", observations.values()),
                          ("pairs.jsonl", pairs)):
        with (output_dir / name).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scenarios": sorted({row["scenario"] for row in rows}),
        "observations": len(observations),
        "cross_camera_pairs": len(pairs),
        "identities": len({row["identity_id"] for row in observations.values()}),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
