"""Quantize the trained encoder with Sony MCT using train-only image crops.

Run in a separate Edge-MDT environment. The resulting ONNX is for imxconv-pt,
not a generic ONNX Runtime INT8 model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from on_sensor_handoff.reid.model import load_checkpoint, observation_input


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "on_sensor_handoff/artifacts/identity_b0_64.pt")
    parser.add_argument("--observations", type=Path,
                        default=ROOT / "on_sensor_handoff/data/aicity_train_v1/observations.jsonl")
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "on_sensor_handoff/artifacts/identity_b0_64_mct.onnx")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--tpc-version", default="1.0",
                        help="IMX500 target platform version supported by the installed converter")
    parser.add_argument("--seed", type=int, default=541)
    args = parser.parse_args()
    if args.samples < 8:
        parser.error("use at least 8 real calibration observations")
    try:
        import model_compression_toolkit as mct
    except ImportError as error:
        raise SystemExit("Install edge-mdt[pt] in a separate converter environment") from error
    rows = [json.loads(line) for line in args.observations.read_text().splitlines()
            if line.strip()]
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("calibration observations must be train-only")
    selected = random.Random(args.seed).sample(rows, min(args.samples, len(rows)))

    def representative_data_gen():
        for row in selected:
            image = observation_input(row, args.base_dir)
            yield [np.transpose(image, (2, 0, 1))[None].astype(np.float32)]

    model = load_checkpoint(args.checkpoint)
    # Current MCT takes a TPC version and device type. Older Edge-MDT releases
    # exposed the same capability through the separate edgemdt_tpc package.
    if hasattr(mct, "get_target_platform_capabilities"):
        tpc = mct.get_target_platform_capabilities(
            tpc_version=args.tpc_version, device_type="imx500")
    else:
        from edgemdt_tpc import get_target_platform_capabilities
        tpc = get_target_platform_capabilities(
            tpc_version=args.tpc_version, device_type="imx500")
    quantized, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=model, representative_data_gen=representative_data_gen,
        target_platform_capabilities=tpc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mct.exporter.pytorch_export_model(
        model=quantized, save_model_path=str(args.output),
        repr_dataset=representative_data_gen, output_names=["embedding"])
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_manifest_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
        "calibration_samples": len(selected),
        "calibration_split": "train",
        "tpc_version": args.tpc_version,
        "quantized_onnx_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "mct_version": getattr(mct, "__version__", "unknown"),
    }
    args.output.with_suffix(".quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
