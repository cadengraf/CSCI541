"""Export a trained static-shape FP32 identity encoder and check parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from on_sensor_handoff.reid.model import EMBEDDING_SIZE, INPUT_SIZE, load_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "on_sensor_handoff/artifacts/identity_b0_64.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "on_sensor_handoff/artifacts/identity_b0_64_fp32.onnx")
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(541)
    sample = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.inference_mode():
        reference = model(sample).numpy()
        torch.onnx.export(model, sample, str(args.output), opset_version=17,
                          input_names=["image_rgb01"], output_names=["embedding"],
                          do_constant_folding=True)
    onnx.checker.check_model(str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.output), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
    if actual.shape != (1, EMBEDDING_SIZE):
        raise ValueError(f"wrong ONNX output shape: {actual.shape}")
    maximum_error = float(np.max(np.abs(reference - actual)))
    if maximum_error > 1e-3:
        raise ValueError(f"PyTorch/ONNX parity failed: max error {maximum_error}")
    contract = {
        "schema_version": 1,
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_sha256": sha256(args.output),
        "input": {"name": "image_rgb01", "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
                  "color": "RGB", "range": [0.0, 1.0]},
        "output": {"name": "embedding", "shape": [1, EMBEDDING_SIZE],
                   "normalization": "L2 on host after tensor read"},
        "preprocessing": "crop ROI, resize 160x160 bilinear; ImageNet normalization inside model",
        "max_torch_onnx_abs_error": maximum_error,
    }
    args.output.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
