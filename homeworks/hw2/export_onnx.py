# CSCI 441/541 — Project 2 · ONNX sample code (Lite CIFAR)
#
# This file is PROVIDED. It does three things and nothing else:
#   1. load_lite_cifar()  — read the training ZIP into tensors
#   2. export_onnx()      — write a model in the exact server contract
#   3. check_onnx()       — check an .onnx file locally before you submit
#
# Your models and training loops are YOURS (generate them with opencode from the
# Part 1 and Part 2 specs, then from your own Part 3 design). Import this file:
#
#     from export_onnx import load_lite_cifar, export_onnx, check_onnx
#     X, y = load_lite_cifar("pj2_lite_cifar_train.zip")   # [n,3,64,64], [n]
#     ... build + train model ...
#     export_onnx(model, "baseline_cnn.onnx")
#     check_onnx("baseline_cnn.onnx", part="part1", images=X[:500], labels=y[:500])
#
# Command line:
#     python export_onnx.py --check baseline_cnn.onnx --part part1 --data pj2_lite_cifar_train.zip
#     python export_onnx.py --demo     # exports an UNTRAINED toy CNN to prove the pipeline works
#
# Server contract (anything else is rejected at submission):
#   input   "image"  float32 [1, 3, 64, 64]  RGB, NCHW, values in [0, 1]
#   output  "logits" float32 [1, 10]          argmax = predicted class
#   Any normalisation (mean/std) must happen INSIDE forward() — the server
#   feeds raw [0, 1] pixels.
#   total float weights <= 3,000,000; opset 17; no dynamic axes.

import importlib.util
import io
import json
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

for _pkg in ("onnx", "onnxruntime"):                 # Colab ships neither
    if importlib.util.find_spec(_pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])

import numpy as np
import torch
from PIL import Image

CLASSES = ["airplane", "automobile", "frog", "horse", "ship"]
INPUT_SHAPE = (1, 3, 64, 64)
MAX_PARAMS = 3_000_000


def load_lite_cifar(path, with_meta=False):
    """Training ZIP or an unzipped folder -> (images, labels[, meta]).

    images: torch.float32 [n, 3, 64, 64] in [0, 1]   labels: torch.int64 [n]
    meta:   the manifest rows
    """
    path = Path(path)
    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path)
        top = next(n for n in zf.namelist() if n.endswith("manifest.json"))[:-len("manifest.json")]
        read = lambda name: zf.read(f"{top}{name}")
    else:
        root = path / "train" if (path / "train").is_dir() else path
        read = lambda name: (root / name).read_bytes()
    manifest = json.loads(read("manifest.json"))
    images = np.stack([
        np.asarray(Image.open(io.BytesIO(read(m["file"]))).convert("RGB"), np.uint8)
        for m in manifest])
    labels = np.array([m["label"] for m in manifest], np.int64)
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    return (x, torch.from_numpy(labels), manifest) if with_meta else (x, torch.from_numpy(labels))


def export_onnx(model, path):
    """Write `model` in the server contract. Do not change this function."""
    model = model.cpu().eval()
    with torch.no_grad():
        out = model(torch.zeros(INPUT_SHAPE))
    assert tuple(out.shape) == (1, len(CLASSES)), \
        f"forward() must return logits of shape (1, 10); got {tuple(out.shape)}"
    with warnings.catch_warnings():               # the TorchScript exporter is the contract
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(model, torch.zeros(INPUT_SHAPE), str(path),
                          input_names=["image"], output_names=["logits"],
                          opset_version=17, dynamo=False)
    print(f"wrote {path}  ({sum(p.numel() for p in model.parameters()):,} parameters)")


def check_onnx(path, part=None, images=None, labels=None):
    """Local pre-flight: ONNX checker, I/O shape, parameter budget, op list,
    and (optionally) accuracy on a labelled batch with onnxruntime.

    The server additionally compares Part 1 and Part 2 models with their specs.
    """
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    assert dims == list(INPUT_SHAPE), f"input shape {dims}, expected {list(INPUT_SHAPE)}"
    params = sum(numpy_helper.to_array(t).size for t in m.graph.initializer
                 if t.data_type == onnx.TensorProto.FLOAT)
    assert params <= MAX_PARAMS, f"{params:,} float weights > {MAX_PARAMS:,} budget"
    ops = sorted({n.op_type for n in m.graph.node})
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    print(f"{path}: OK  input {dims}  params {params:,}  convs {len(convs)}")
    print(f"  ops: {', '.join(ops)}")
    gpool = any(n.op_type in ("GlobalAveragePool", "GlobalMaxPool") for n in m.graph.node)
    if part == "part1" and gpool:
        print("  WARNING: Part 1 is the Flatten baseline - it must not contain global pooling")
    if part == "part2" and not any(n.op_type == "GlobalAveragePool" for n in m.graph.node):
        print("  WARNING: Part 2 needs nn.AdaptiveAvgPool2d(1) (exports as GlobalAveragePool) before the classifier")

    if images is not None and labels is not None:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        to_np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        x, y = to_np(images), to_np(labels)
        preds = np.array([sess.run(None, {"image": x[i:i + 1].astype(np.float32)})[0].argmax()
                          for i in range(len(x))])
        print(f"  onnxruntime accuracy on {len(x)} given images: {(preds == y).mean():.3f}")
    return params


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", help="an .onnx file to verify")
    ap.add_argument("--part", choices=["part1", "part2", "part3"])
    ap.add_argument("--data", help="training ZIP for --check (accuracy on 500 images)")
    ap.add_argument("--demo", action="store_true", help="export an untrained toy CNN")
    args, _unknown = ap.parse_known_args()        # Colab injects -f kernel.json

    if args.demo:
        toy = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(4),
            torch.nn.Flatten(), torch.nn.Linear(8 * 16 * 16, len(CLASSES)))
        export_onnx(toy, "toy_untrained.onnx")
        check_onnx("toy_untrained.onnx")
    elif args.check:
        X = y = None
        if args.data:
            X, y = load_lite_cifar(args.data)
            pick = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:500]
            X, y = X[pick], y[pick]
        check_onnx(args.check, args.part, X, y)
    else:
        ap.print_help()
