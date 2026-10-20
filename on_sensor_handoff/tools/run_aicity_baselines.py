"""Compare the existing Pi fingerprint and pretrained FastViT on one gallery.

Ground-truth boxes provide all candidates. This measures appearance matching,
not candidate discovery, tracking, IMX500 inference, or a trained ReID head.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_fingerprint(source: Path, include: Path, output: Path) -> ctypes.CDLL:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "g++", "-std=c++17", "-O2", "-fPIC", "-shared",
        "-I", str(include), str(Path(__file__).with_name("fingerprint_bridge.cpp")),
        str(source), "-o", str(output),
    ], check=True)
    library = ctypes.CDLL(str(output))
    library.fingerprint_yuv420.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    library.fingerprint_yuv420.restype = ctypes.c_int
    return library


@lru_cache(maxsize=4)
def read_frame(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"cannot read {path}")
    return frame


def fastvit_input(image: np.ndarray, box: list[int]) -> np.ndarray:
    x, y, w, h = box
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"empty crop: {box}")
    # Mirrors seg_tracking's EfficientNetService preprocessing for non-V-JEPA.
    resize_side = math.floor(256 / 0.9 + 0.5)
    scale = resize_side / min(crop.shape[:2])
    new_size = (max(resize_side, math.floor(crop.shape[1] * scale + 0.5)),
                max(resize_side, math.floor(crop.shape[0] * scale + 0.5)))
    resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_CUBIC)
    left = (resized.shape[1] - 256) // 2
    top = (resized.shape[0] - 256) // 2
    rgb = cv2.cvtColor(resized[top:top + 256, left:left + 256], cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32)
    return np.transpose(rgb, (2, 0, 1))[None].copy()


def identity_input(image: np.ndarray, box: list[int]) -> np.ndarray:
    x, y, w, h = box
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"empty crop: {box}")
    resized = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None].copy()


def fingerprint(image: np.ndarray, box: list[int], library: ctypes.CDLL) -> np.ndarray:
    height, width = image.shape[:2]
    if width % 2 or height % 2:
        raise ValueError("fingerprint expects even YUV420 frame dimensions")
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    yuv = np.ascontiguousarray(yuv)
    output = np.empty(140, np.float32)
    x, y, w, h = box
    count = library.fingerprint_yuv420(
        yuv.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        width, height, width, x, y, w, h,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 140)
    if count != 140:
        raise ValueError(f"fingerprint failed for box {box}: code {count}")
    return output


def score_trials(trials: list[dict], observations: dict[str, dict],
                 features: dict[str, np.ndarray]) -> list[dict]:
    scored = []
    for trial in trials:
        query = features[trial["query"]]
        candidates = trial["candidates"]
        scores = [float(query @ features[key]) for key in candidates]
        winner_index = int(np.argmax(scores))
        winner = candidates[winner_index]
        scored.append({
            "trial_id": trial["trial_id"],
            "split": trial["split"],
            "present": trial["present"],
            "candidate_count": len(candidates),
            "top_score": scores[winner_index],
            "top_identity_id": observations[winner]["identity_id"],
            "target_identity_id": trial["target_identity_id"],
            "rank1_correct": observations[winner]["identity_id"] == trial["target_identity_id"],
        })
    return scored


def metrics(scored: list[dict], threshold: float) -> dict:
    present = [row for row in scored if row["present"]]
    absent = [row for row in scored if not row["present"]]
    accepted_present = [row for row in present if row["top_score"] >= threshold]
    accepted_absent = [row for row in absent if row["top_score"] >= threshold]
    correct = sum(row["rank1_correct"] for row in accepted_present)
    return {
        "present_trials": len(present),
        "absent_trials": len(absent),
        "mean_candidates": sum(row["candidate_count"] for row in scored) / len(scored),
        "present_rank1_correct": sum(row["rank1_correct"] for row in present),
        "correct_handoffs": correct,
        "wrong_identity_handoffs": len(accepted_present) - correct,
        "missed_present": len(present) - len(accepted_present),
        "false_accept_absent": len(accepted_absent),
        "correct_reject_absent": len(absent) - len(accepted_absent),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "on_sensor_handoff/data/aicity_smoke_v1")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fingerprint-source", type=Path, required=True)
    parser.add_argument("--fingerprint-include", type=Path, required=True)
    parser.add_argument("--identity-model", type=Path,
                        help="optional trained 160x160 FP32 ONNX encoder")
    parser.add_argument("--quantized-model", type=Path,
                        help="optional MCT-exported ONNX encoder")
    parser.add_argument("--sensor-embeddings", type=Path,
                        help="optional captured IMX500 vectors keyed by observation_id")
    parser.add_argument("--protocol", default="GT candidate gallery; cosine top-1; validation absent-max threshold")
    parser.add_argument("--output", type=Path, default=ROOT / "on_sensor_handoff/runs/aicity_smoke_v1.json")
    args = parser.parse_args()
    observations = {row["observation_id"]: row for row in load_jsonl(args.data_dir / "observations.jsonl")}
    trials = [row for row in load_jsonl(args.data_dir / "trials.jsonl")
              if row["split"] in ("validation", "test")]
    required = {trial["query"] for trial in trials}
    required.update(key for trial in trials for key in trial["candidates"])
    shared_library = args.output.parent / "fingerprint_bridge.so"
    library = build_fingerprint(args.fingerprint_source, args.fingerprint_include, shared_library)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    if session.get_inputs()[0].shape[1:] != [3, 256, 256]:
        raise ValueError("this benchmark expects a 256x256 RGB encoder")
    features = {"fingerprint": {}, "fastvit": {}}
    identity_session = None
    quantized_session = None
    if args.identity_model:
        identity_session = ort.InferenceSession(str(args.identity_model),
                                                sess_options=options,
                                                providers=["CPUExecutionProvider"])
        if identity_session.get_inputs()[0].shape != [1, 3, 160, 160]:
            raise ValueError("trained identity encoder needs fixed [1,3,160,160] input")
        features["identity_fp32"] = {}
    if args.quantized_model:
        try:
            import mct_quantizers as mctq
        except ImportError as error:
            raise SystemExit("MCT ONNX evaluation needs mct_quantizers") from error
        quantized_session = ort.InferenceSession(
            str(args.quantized_model), sess_options=mctq.get_ort_session_options(),
            providers=["CPUExecutionProvider"])
        if quantized_session.get_inputs()[0].shape != [1, 3, 160, 160]:
            raise ValueError("quantized identity encoder needs fixed [1,3,160,160] input")
        features["identity_mct_onnx"] = {}
    for index, key in enumerate(sorted(required), 1):
        row = observations[key]
        frame = read_frame(str(args.base_dir / row["image_path"]))
        box = row["bbox_xywh"]
        features["fingerprint"][key] = fingerprint(frame, box, library)
        vector = session.run(None, {input_name: fastvit_input(frame, box)})[0][0]
        features["fastvit"][key] = vector / max(float(np.linalg.norm(vector)), 1e-8)
        if identity_session is not None or quantized_session is not None:
            encoded_input = identity_input(frame, box)
        if identity_session is not None:
            name = identity_session.get_inputs()[0].name
            vector = identity_session.run(None, {name: encoded_input})[0][0]
            features["identity_fp32"][key] = vector / max(float(np.linalg.norm(vector)), 1e-8)
        if quantized_session is not None:
            name = quantized_session.get_inputs()[0].name
            vector = quantized_session.run(None, {name: encoded_input})[0][0]
            features["identity_mct_onnx"][key] = vector / max(float(np.linalg.norm(vector)), 1e-8)
        if index % 25 == 0 or index == len(required):
            print(f"encoded {index}/{len(required)} observations", flush=True)

    sensor_model_hash = None
    if args.sensor_embeddings:
        captured = load_jsonl(args.sensor_embeddings)
        sensor_model_hashes = {row["model_sha256"] for row in captured}
        if len(sensor_model_hashes) != 1:
            raise ValueError("sensor embeddings must come from one RPK version")
        sensor_model_hash = next(iter(sensor_model_hashes))
        vectors = {}
        for row in captured:
            key = row["observation_id"]
            if key in vectors:
                raise ValueError(f"duplicate sensor embedding: {key}")
            vector = np.asarray(row["embedding"], dtype=np.float32)
            if vector.shape != (64,) or not np.isfinite(vector).all():
                raise ValueError(f"invalid sensor embedding: {key}")
            vectors[key] = vector / max(float(np.linalg.norm(vector)), 1e-8)
        missing = required - vectors.keys()
        if missing:
            raise ValueError(f"missing {len(missing)} required sensor embeddings")
        features["identity_imx500"] = vectors

    results = {}
    for name, vectors in features.items():
        scored = score_trials(trials, observations, vectors)
        calibration_absent = [row["top_score"] for row in scored
                              if row["split"] == "validation" and not row["present"]]
        if not calibration_absent:
            raise ValueError("no absent validation trials for threshold calibration")
        # Small smoke set: choose a conservative threshold with zero *observed*
        # false accepts on validation. It does not guarantee a population FAR.
        threshold = math.nextafter(max(calibration_absent), math.inf)
        results[name] = {
            "threshold": threshold,
            "validation": metrics([row for row in scored if row["split"] == "validation"], threshold),
            "test": metrics([row for row in scored if row["split"] == "test"], threshold),
            "trials": scored,
        }
    report = {
        "protocol": args.protocol,
        "model_sha256": file_hash(args.model),
        "fingerprint_source_sha256": file_hash(args.fingerprint_source),
        "identity_model_sha256": file_hash(args.identity_model) if args.identity_model else None,
        "quantized_model_sha256": file_hash(args.quantized_model) if args.quantized_model else None,
        "sensor_rpk_sha256": sensor_model_hash,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(name, "threshold", round(result["threshold"], 4), "test", result["test"])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
