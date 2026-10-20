"""Capture actual IMX500 identity tensors and matching camera frames on a Pi.

Run this on one AI Camera at a time after packaging the model as an RPK. The
object ROI is manually supplied in full 4056x3040 sensor coordinates for this
first appearance-only experiment. Candidate discovery is a later stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def safe_token(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if result in ("", ".", ".."):
        raise ValueError(f"invalid path token {value!r}")
    return result


def parse_vector(outputs: list[np.ndarray] | None, expected_size: int = 64) -> np.ndarray | None:
    if outputs is None or len(outputs) != 1:
        return None
    vector = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    if (vector.size != expected_size or not np.isfinite(vector).all() or
            np.linalg.norm(vector) <= 1e-8):
        return None
    return vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="packaged identity network.rpk")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--capture-group", required=True,
                        help="recording group shared across both cameras")
    parser.add_argument("--scene-id", required=True,
                        help="one target query and its destination candidate pool")
    parser.add_argument("--identity-id", required=True,
                        help="physical object label consistent across cameras")
    parser.add_argument("--category", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--role", choices=("query", "candidate"), required=True)
    parser.add_argument("--roi", type=int, nargs=4, required=True,
                        metavar=("X", "Y", "W", "H"),
                        help="box in full IMX500 sensor pixel coordinates")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "on_sensor_handoff/data/live_sensor_v1")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.count < 1 or args.warmup < 0:
        parser.error("count must be >= 1 and warmup >= 0")
    x, y, w, h = args.roi
    if min(x, y) < 0 or min(w, h) < 8 or x + w > 4056 or y + h > 3040:
        parser.error("ROI must be an 8+ pixel box inside the 4056x3040 sensor")
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.relative_to(base_dir)
    model_sha = hashlib.sha256(args.model.read_bytes()).hexdigest()

    # Import here so --help and local format tests work off the Raspberry Pi.
    from picamera2 import Picamera2
    from picamera2.devices import IMX500

    imx500 = IMX500(str(args.model))  # before Picamera2 instantiation
    if tuple(imx500.get_input_size()) != (160, 160):
        raise ValueError(f"wrong RPK input size: {imx500.get_input_size()}")
    input_tensor_config = imx500.config.get("input_tensor", {})
    input_format = input_tensor_config.get("input_format", "RGB")
    if input_format != "RGB":
        raise ValueError(f"RPK input format is {input_format}, but this model expects RGB")
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_video_configuration(
        main={"size": (1280, 960), "format": "RGB888"}, buffer_count=4,
        queue=False)
    picam2.configure(config)
    imx500.set_inference_roi_abs((x, y, w, h))
    applied_roi = imx500.config.get("roi")
    if applied_roi is None or tuple(applied_roi.to_tuple()) != (x, y, w, h):
        raise RuntimeError(f"IMX500 did not apply requested ROI: {args.roi}")
    image_dir = output_dir / "images" / safe_token(args.capture_group) / safe_token(args.camera_id)
    image_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id += f"_{safe_token(args.camera_id)}_{safe_token(args.scene_id)}_{safe_token(args.role)}"
    capture_file = output_dir / f"capture_{run_id}.jsonl"
    recorded = 0
    missing = 0
    consecutive_missing = 0
    recorded_timestamps_ns: list[int] = []
    start = time.monotonic()
    try:
        picam2.start()
        with capture_file.open("w", encoding="utf-8") as stream:
            while recorded < args.count:
                request = picam2.capture_request()
                try:
                    metadata = request.get_metadata()
                    outputs = imx500.get_outputs(metadata)
                    vector = parse_vector(outputs)
                    if vector is None:
                        missing += 1
                        consecutive_missing += 1
                        if consecutive_missing > 300:
                            raise RuntimeError("no valid 64-value IMX500 output after repeated frames")
                        continue
                    consecutive_missing = 0
                    if args.warmup:
                        args.warmup -= 1
                        continue
                    shape = [list(shape) for shape in imx500.get_output_shapes(metadata)]
                    mapped_roi = imx500.get_roi_scaled(request)
                    if hasattr(mapped_roi, "to_tuple"):
                        mapped_roi = mapped_roi.to_tuple()
                    roi_preview = [int(value) for value in mapped_roi]
                    if (roi_preview[0] < 0 or roi_preview[1] < 0 or
                            roi_preview[2] < 8 or roi_preview[3] < 8 or
                            roi_preview[0] + roi_preview[2] > 1280 or
                            roi_preview[1] + roi_preview[3] > 960):
                        raise ValueError(f"invalid preview ROI mapping: {roi_preview}")
                    timestamp_ns = int(metadata.get("SensorTimestamp", time.monotonic_ns()))
                    observation_id = (f"{args.capture_group}/{args.camera_id}/"
                                      f"{timestamp_ns}/{recorded}")
                    image_path = image_dir / f"{timestamp_ns}_{recorded}.jpg"
                    request.save("main", str(image_path))
                    kpi = imx500.get_kpi_info(metadata)
                    row = {
                        "schema_version": 1,
                        "dataset": "paired_ai_cameras",
                        "capture_group": args.capture_group,
                        "scene_id": args.scene_id,
                        "camera_id": args.camera_id,
                        "track_id": args.identity_id,
                        "identity_id": args.identity_id,
                        "category": args.category,
                        "split": args.split,
                        "role": args.role,
                        "timestamp_us": timestamp_ns // 1000,
                        "image_path": str(image_path.relative_to(base_dir)),
                        "bbox_xywh": roi_preview,
                        "observation_id": observation_id,
                        "roi_sensor_xywh": [x, y, w, h],
                        "embedding": vector.tolist(),
                        "embedding_size": vector.size,
                        "input_format": input_format,
                        "input_dtype": input_tensor_config.get("dtype"),
                        "output_shapes": shape,
                        "model_sha256": model_sha,
                        "kpi_ms": list(kpi) if kpi is not None else None,
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    recorded_timestamps_ns.append(timestamp_ns)
                    recorded += 1
                finally:
                    request.release()
    finally:
        picam2.stop()
        picam2.close()
    elapsed = time.monotonic() - start
    inference_rate = None
    if len(recorded_timestamps_ns) > 1:
        span = (recorded_timestamps_ns[-1] - recorded_timestamps_ns[0]) / 1e9
        if span > 0:
            inference_rate = (len(recorded_timestamps_ns) - 1) / span
    print(json.dumps({"capture_file": str(capture_file), "recorded": recorded,
                      "missing_output_frames": missing,
                      "startup_and_capture_seconds": elapsed,
                      "observed_inference_rate_hz": inference_rate}, indent=2))


if __name__ == "__main__":
    main()
