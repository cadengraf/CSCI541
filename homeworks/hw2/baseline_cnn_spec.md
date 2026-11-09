# Part 1 & Part 2 — CNN specifications

Give this file to opencode together with `export_onnx.py` and the training ZIP.
The Evaluation Site checks the exported graph against the **Architecture**
table of the part you submit; a model that deviates is rejected with the reason.
Part 3 has no spec — it is your own design.

## Data (all parts)

- `pj2_lite_cifar_train.zip` — 5,000 labelled 64×64 RGB images, 5 classes
  (`airplane automobile frog horse ship`, label id = index), 1,000 per class.
  Each image is one 32×32 photo pasted in the centre of a grey canvas.
- Load it with `load_lite_cifar()` from `export_onnx.py`: images
  `float32 [n,3,64,64]` in `[0,1]`, labels `int64 [n]`.
- Hold out 10% as a validation split: `torch.randperm(n, generator=torch.Generator().manual_seed(0))`,
  first 500 indices = validation.

## Part 1 architecture — Flatten head

| # | Layer | Output shape |
|---|---|---|
| 0 | input, then normalise inside `forward`: `(x - 0.5) / 0.25` | 3×64×64 |
| 1 | Conv 5×5, 32 filters, stride 1, padding 2 → ReLU → MaxPool 2×2 | 32×32×32 |
| 2 | Conv 3×3, 64 filters, stride 1, padding 1 → ReLU → MaxPool 2×2 | 64×16×16 |
| 3 | Conv 3×3, 128 filters, stride 1, padding 1 → ReLU → MaxPool 2×2 | 128×8×8 |
| 4 | Conv 3×3, 128 filters, stride 1, padding 1 → ReLU | 128×8×8 |
| 5 | Flatten | 8,192 |
| 6 | Linear 8192 → 256 → ReLU → Dropout(0.5) | 256 |
| 7 | Linear 256 → 5 (logits, no softmax) | 5 |

Layers 1–4 are the **backbone** (`self.features`); layers 5–7 are the
**head** (`self.classifier`). Keep them as two separate `nn.Sequential`
attributes. Total: 2,341,061 parameters. No BatchNorm, no extra layers, no
branches.

## Part 2 architecture — global average pooling head

Same backbone (layers 0–4, unchanged). Only the head changes:

| # | Layer | Output shape |
|---|---|---|
| 5 | Global average pooling: `nn.AdaptiveAvgPool2d(1)` → Flatten | 128 |
| 6 | Linear 128 → 256 → ReLU → Dropout(0.5) | 256 |
| 7 | Linear 256 → 5 (logits, no softmax) | 5 |

Total: 276,677 parameters. Everything else — data, split, training, export —
stays exactly as in Part 1, so the head is the only difference between the two
models.

## Training (Parts 1 and 2)

- Loss `nn.CrossEntropyLoss`, optimizer `torch.optim.Adam(lr=1e-3)`, batch 64,
  50 epochs; use CUDA if available. `torch.manual_seed(seed)` before building
  the model, with `seed` a command-line argument (default 0).
- Augmentation: random horizontal flip only. Nothing else.
- Print train loss and validation accuracy every epoch.
- Keep the weights of the epoch with the best validation accuracy.

## Export

- Part 1: `export_onnx(model, "baseline_cnn.onnx")`, then
  `check_onnx("baseline_cnn.onnx", part="part1", images=X_val, labels=y_val)`.
- Part 2: `export_onnx(model, "gap_cnn.onnx")`, then
  `check_onnx("gap_cnn.onnx", part="part2", images=X_val, labels=y_val)`.
- Do not modify `export_onnx.py`.

## Suggested opencode prompt (Part 1)

```text
Read baseline_cnn_spec.md and export_onnx.py. Write train_baseline.py that
implements the Part 1 baseline CNN exactly as specified (architecture table,
training settings), trains it on pj2_lite_cifar_train.zip, and exports
baseline_cnn.onnx with export_onnx() followed by check_onnx(). Use the helper
functions from export_onnx.py; do not edit that file. Then run the script and
show me the validation accuracy and the check_onnx output.
```

For Part 2, write the prompt yourself: tell opencode what must change (the
head) and what must not (everything else), then check its diff.
