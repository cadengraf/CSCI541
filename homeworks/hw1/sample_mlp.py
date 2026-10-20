"""Self-contained script for the linear baseline and final two-hidden-layer MLP."""

import io
import json
import zipfile
import numpy as np
from PIL import Image
import importlib.util, subprocess, sys
if importlib.util.find_spec("onnx") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx"])
import onnx
from onnx import helper, TensorProto

GLYPHS = "〇一二三四五六七八九" + "十"      # class order = label ids 0..10
N_CLASSES = 11
INPUT_SHAPE = (1, 1, 64, 64)               # model contract: NCHW float32 [0,1]
def load_copybook(dataset_path):
    """Accepts a Copybook export zip or an unpacked directory.
    Returns (images float32 [n,64,64] in [0,1], labels int64 [n])."""
    if dataset_path.endswith(".zip"):
        zf = zipfile.ZipFile(dataset_path)
        meta = json.loads(zf.read("labels.json") if "labels.json" in zf.namelist()
                          else zf.read("manifest.json"))
        if isinstance(meta, dict):                          # copybook layout
            meta = meta.get("labels") or meta.get("shots")
        shots = []
        for m in meta:                                      # manfifest order
            name = f"shots/{m['file']}"
            img = Image.open(io.BytesIO(zf.read(name))).convert("L")
            shots.append(np.asarray(img, np.float32) / 255.0)
        return (np.stack(shots),
                np.array([m["label"] for m in meta], dtype=np.int64))
    raise SystemExit("point --dataset at your Copybook export zip")

def softmax(z):
    """Stable softmax (lecture 7 recipe). Provided."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

D_A = 16 * 16          # representation width: 16x16 average-pooled grayscale
def part_a_features(images):
    """64x64 -> 16x16 average pool, ink positive, scaled to [-1,1]."""
    x = images.reshape(-1, 16, 4, 16, 4).mean(axis=(2, 4))
    return ((0.5 - x) * 2.0).reshape(len(images), -1)   # ink dark: invert
def part_a_train(X, y, epochs=60, lr=0.5):
    n, d = X.shape
    W = np.zeros((d, N_CLASSES), np.float32)    # zero init: sanity check below
    b = np.zeros(N_CLASSES, np.float32)

    n_check = min(64, n)   # works even with a tiny first dataset
    p0 = softmax(X[:n_check] @ W + b)
    loss0 = -np.log(p0[np.arange(n_check), y[:n_check]] + 1e-12).mean()
    assert abs(loss0 - np.log(N_CLASSES)) < 1e-3, "softmax/CE sanity failed"
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(n)
        for i in range(0, n, 64):
            j = perm[i:i + 64]
            Xb, yb = X[j], y[j]
            # ▼ blank A1 — affine forward · Lecture 7
            logits = Xb @ W + b
            p = softmax(logits)

            # ▼ blank A2 — softmax cross-entropy gradient · Lecture 7
            p[np.arange(len(yb)), yb] -= 1.0
            grad_W = Xb.T @ p / len(yb)
            grad_b = p.mean(axis=0)
            # ▼ blank A3 — SGD update · Lecture 8
            W -= lr * grad_W
            b -= lr * grad_b
            
        if ep % 20 == 0 or ep == epochs - 1:
            acc = (softmax(X @ W + b).argmax(1) == y).mean()
            print(f"[A] epoch {ep:3d}  train acc {acc:.3f}")
    return W, b
def export_linear(W, b, path="linear_model.onnx"):

    def tensor(name, arr):
        return helper.make_tensor(name, TensorProto.FLOAT, arr.shape,
                                  arr.astype(np.float32).tobytes(), raw=True)
    shape = helper.make_tensor("fshape", TensorProto.INT64, [2],
                               np.array([1, D_A], np.int64).tobytes(), raw=True)
    nodes = [
        helper.make_node("AveragePool", ["image"], ["p"], kernel_shape=[4, 4], strides=[4, 4]),
        helper.make_node("Constant", [], ["half"],
                         value=tensor("half", np.array(0.5, np.float32))),
        helper.make_node("Sub", ["half", "p"], ["inv"]),          # 0.5 - pool
        helper.make_node("Constant", [], ["two"],
                         value=tensor("two", np.array(2.0, np.float32))),
        helper.make_node("Mul", ["inv", "two"], ["x"]),           # scale
        helper.make_node("Reshape", ["x", "fshape"], ["f"]),      # [1, 256]
        helper.make_node("Gemm", ["f", "Wt", "bt"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "linear_softmax",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, INPUT_SHAPE)],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT,
                                       [1, N_CLASSES])],
        [tensor("Wt", W.astype(np.float32)),
         tensor("bt", b), shape])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"[A] wrote {path}  ({W.size + b.size} params) — submit for the Assignment")

# — PART B —
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json

BUDGET = 50_000
def part_b_features(images):
    """images [n,64,64] in [0,1]  ->  features [n, d], bounded."""
    # Average each 8x8 block to keep broader stroke shapes.
    # ▼ blank B1 — representation · New idea · this project

    pooled = images.reshape(-1, 8, 8, 8, 8).mean(axis=(2, 4))
    return (1.0 - pooled).reshape(len(images), -1)

def count_params(model):
    # ▼ blank B2 — parameter accounting · Lecture 9

    return sum(p.numel() for p in model.parameters())



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # ▼ blank B3a — declare your layers · Lecture 9
        self.fc1 = nn.Linear(8 * 8, 256)
        self.fc2 = nn.Linear(256, 96)
        self.fc3 = nn.Linear(96, 11)   # N_CLASSES = 11
    def forward(self, x):                     # x: [1,1,64,64]
        # ▼ blank B3b — forward: preprocessing in-graph + layers · Lecture 9

        x = F.avg_pool2d(x, kernel_size=8)
        x = (1.0 - x).reshape(x.shape[0], -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
        
def init_weights(m):
    # ▼ blank B4 — initialization policy · Lecture 9

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
        m.bias.data.fill_(0.01)

def part_b_train(model, images, y, imgs_val, yval, epochs=200, lr=0.1, batch=32):
    """Feed RAW images [n,1,64,64]: the model preprocesses inside forward,
    exactly like the exported graph will (Lecture 9 encapsulation)."""
    X = torch.from_numpy(images.reshape(-1, 1, 64, 64))
    XvT = torch.from_numpy(imgs_val.reshape(-1, 1, 64, 64))
    yt = torch.from_numpy(y)
    opt = torch.optim.SGD(model.parameters(), lr=lr)   # Lecture 9 optimizer
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            # ▼ blank B5 — the four-step training step · Lecture 9

            opt.zero_grad()
            logits = model(X[j])
            loss = loss_fn(logits, yt[j])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(XvT).argmax(1)
                       == torch.from_numpy(yval)).float().mean().item()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"[B] epoch {ep:3d}  val acc {val_acc:.3f}")
def export_mlp(model, path="mlp_model.onnx"):
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 1, 64, 64), path,
                      input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
# — run —
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/megrad/Documents/GitHub/CSCI541/pj1_combined_10aug.zip", help="your Copybook export")
    ap.add_argument("--val", default=None, help="optional manifest dir, e.g. the public set")
    args, _unknown = ap.parse_known_args()   # Colab injects -f kernel.json; ignore it
    images, labels = load_copybook(args.dataset)
    print(f"loaded {len(images)} traced images "
          f"({dict(zip(*np.unique(labels, return_counts=True)))})")
# — Part A —
    XA = part_a_features(images)
    W, b = part_a_train(XA, labels)
    export_linear(W, b)
# — Part B —
    XB = part_b_features(images)                       # TODO-B1 prototype
    assert XB.ndim == 2 and np.abs(XB).max() <= 1.0 + 1e-5, \
        "representation must be a bounded [n, d] matrix"
    print(f"[B] representation prototype: {XB.shape[1]} dims "
          f"(mirror the SAME preprocessing inside MLP.forward, TODO-B3)")
    if args.val:                                        # public set as validation
        vm = json.load(open(f"{args.val}/manifest.json"))
        vi = np.stack([np.asarray(Image.open(f"{args.val}/{m['file']}").convert('L'),
                                  np.float32) / 255.0 for m in vm])
        vl = np.array([m["label"] for m in vm], np.int64)
    else:
        cut = max(8, len(images) // 5)
        vi, vl = images[:cut], labels[:cut]
    model = MLP()                                       # TODO-B3
    model.apply(init_weights)                           # TODO-B4
    n_params = count_params(model)                      # TODO-B2
    assert n_params <= BUDGET, f"{n_params} params exceeds the {BUDGET} budget"
    print(f"[B] architecture accepted: {n_params} parameters "
          f"({100 * n_params / BUDGET:.0f}% of budget)")
    part_b_train(model, images, labels, vi, vl)         # TODO-B5
    export_mlp(model)                                   # provided
    import onnx
    m = onnx.load("mlp_model.onnx")
    onnx.checker.check_model(m)
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 1, 64, 64], f"input shape {dims} — export block was modified?"
    print("[B] mlp_model.onnx verified — submit at /board.html and keep the "
          "submission ID for your report")
