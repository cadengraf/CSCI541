"""Import a frozen AICity22 vehicle handoff subset into the observation contract.

The source videos and annotations are read locally. Extracted frames stay in
the ignored data directory; only the split definition belongs in Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]


def load_gt(path: Path) -> dict[int, dict[int, list[int]]]:
    frames: dict[int, dict[int, list[int]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            values = line.strip().split(",")
            if len(values) < 6:
                continue
            frame, identity = int(values[0]), int(values[1])
            box = [round(float(value)) for value in values[2:6]]
            if box[2] >= 8 and box[3] >= 8:
                frames[frame][identity] = box
    return frames


def observation_id(camera: str, frame: int, identity: int) -> str:
    return f"{camera}/{frame}/{identity}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "on_sensor_handoff/data/aicity_smoke_v1")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    base = args.base_dir.resolve()
    output = args.output_dir.resolve()
    output.relative_to(base)  # image_path is portable relative to this directory

    raw = args.source_manifest.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    frozen = json.loads(args.split.read_text(encoding="utf-8"))
    if frozen["source_manifest_sha256"] != source_sha:
        raise ValueError("source benchmark manifest hash differs from frozen split")
    trials_by_id = {row["trial_id"]: row for row in map(json.loads, raw.splitlines())}
    selected = []
    for split, present_ids in frozen["splits"].items():
        for present_id in present_ids:
            for trial_id in (present_id, present_id + "-absent"):
                row = trials_by_id[trial_id]
                if row["present"] != (trial_id == present_id):
                    raise ValueError(f"unexpected present flag: {trial_id}")
                selected.append((split, row))
    if len({row["trial_id"] for _, row in selected}) != len(selected):
        raise ValueError("trial reused across splits")
    scenario_splits: dict[str, str] = {}
    for split, row in selected:
        old = scenario_splits.setdefault(row["scenario"], split)
        if old != split:
            raise ValueError(f"scenario leaks across splits: {row['scenario']}")

    gt_cache: dict[Path, dict[int, dict[int, list[int]]]] = {}
    video_cache: dict[Path, cv2.VideoCapture] = {}
    observations: dict[str, dict] = {}
    imported_trials = []

    def frame_image(video_path: Path, camera: str, frame: int, split: str):
        image_path = output / "frames" / split / camera / f"{frame:06d}.jpg"
        if image_path.exists():
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"cannot decode {image_path}")
        else:
            capture = video_cache.get(video_path)
            if capture is None:
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise ValueError(f"cannot open video {video_path}")
                video_cache[video_path] = capture
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"cannot read {video_path} frame {frame}")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError(f"cannot write {image_path}")
        return image, str(image_path.relative_to(base))

    def add_observation(split: str, row: dict, camera: str, video: Path,
                        frame: int, identity: int, box: list[int]) -> str:
        key = observation_id(camera, frame, identity)
        if key in observations:
            return key
        image, image_path = frame_image(video, camera, frame, split)
        height, width = image.shape[:2]
        x, y, w, h = box
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError(f"invalid clipped box {key}: {box} on {width}x{height}")
        observations[key] = {
            "schema_version": 1,
            "dataset": "AICity22_Track1_MTMC_Tracking",
            "capture_group": row["scenario"],
            "camera_id": camera,
            "track_id": str(identity),
            "identity_id": f"{row['scenario']}/{identity}",
            "category": "vehicle",
            "timestamp_us": round((frame - 1) * 100000),
            "image_path": image_path,
            "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
            "observation_id": key,
            "frame_index": frame,
            "split": split,
        }
        return key

    try:
        for split, row in selected:
            query_sample = row["source_samples"][-1]
            source_box = query_sample["bbox_xyxy"]
            source_xywh = [round(source_box[0]), round(source_box[1]),
                           round(source_box[2] - source_box[0]),
                           round(source_box[3] - source_box[1])]
            query = add_observation(split, row, row["source_camera"],
                                    Path(row["source_video"]), query_sample["frame"],
                                    row["target_id"], source_xywh)
            gt_path = Path(row["destination_gt"])
            if gt_path not in gt_cache:
                gt_cache[gt_path] = load_gt(gt_path)
            candidate_boxes = gt_cache[gt_path].get(row["destination_frame"], {})
            present = row["target_id"] in candidate_boxes
            if present != row["present"]:
                raise ValueError(f"GT presence differs from benchmark: {row['trial_id']}")
            if not candidate_boxes:
                raise ValueError(f"empty destination gallery: {row['trial_id']}")
            candidates = [add_observation(split, row, row["destination_camera"],
                                          Path(row["destination_video"]),
                                          row["destination_frame"], identity, box)
                          for identity, box in sorted(candidate_boxes.items())]
            imported_trials.append({
                "trial_id": row["trial_id"],
                "split": split,
                "query": query,
                "candidates": candidates,
                "target_identity_id": f"{row['scenario']}/{row['target_id']}",
                "present": row["present"],
            })
    finally:
        for capture in video_cache.values():
            capture.release()

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("observations.jsonl", observations.values()),
                       ("trials.jsonl", imported_trials)):
        with (output / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(observations)} observations and {len(imported_trials)} trials to {output}")


if __name__ == "__main__":
    main()
