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
        for m in meta:                                      # manifest order
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
