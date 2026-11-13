# Matched training with aspect versus square identity crops

## Run

From `capstone/reid`, with the PC virtual environment active:

```bash
python train.py compare-crops
```

For SSH:

```bash
tmux new -s crop-training
python train.py compare-crops
```

Detach with Ctrl+B, then D; reattach with `tmux attach -t crop-training`.
Repeat the Python command to resume. Each arm checkpoints after 100 complete
cycles and at epoch boundaries. A cycle has two optimizer updates. Completed
arms are skipped; do not run two writers on these paths.

Settings: `configs/crop_training.yaml`. The command runs `aspect_384`, then
`square_256`, then evaluates each selected checkpoint on calibration and KITTI,
and cross-evaluates the selected weights under both inference crop policies.
The long run is not started by setup/checks.

```bash
python train.py compare-crops-check
python benchmark.py compare-crops  # reevaluate completed training
```

## What changes

| Property | aspect_384 | square_256 |
| --- | --- | --- |
| Identity-crop geometry | Preserve aspect ratio | Stretch to square |
| Identity-crop size | Longest side capped at 384 | 256 × 256 |
| VRAI training crops | Aspect policy | Square policy |
| Own-policy calibration | Aspect policy | Square policy |
| KITTI full-frame resize | Existing short side 384 / max side 1280 | Identical |
| Architecture | Existing joint Co-DINO/RefineMask/ReID | Identical; no spatial adapter |

Only `_prepare(..., crop=True)` uses crop geometry. Full-frame box/mask/identity
training and full-frame inference retain the same preprocessing. The model is
still one shared backbone with box/mask/identity branches; this is a preprocessing
experiment, not a detector/backbone upgrade. Square-crop training affects the
VRAI identity update; KITTI identity features come from unwarped full-frame ROIs.

Both arms start from the pinned latest distilled joint checkpoint and retain
joint KITTI detection/mask/identity losses plus VRAI classification/triplet
losses. New teacher distillation is off. The architecture and loaded weights are
identical at epoch zero, but their crop inference scores differ immediately due
to preprocessing. Four epochs × 1,000 cycles give 8,000 scheduled updates per arm,
with identical pairs, identity samples, augmentation seeds, optimizer settings,
and four trainable backbone blocks. Image geometry changes compute per update;
this is an equal-update comparison, not an equal-FLOPs comparison.

## Evaluation

Full VRAI calibration selects each arm's best checkpoint using its own crop
policy; epoch zero remains an eligible fallback. The comparison report includes
initial-to-final mAP change, selected epoch and skipped updates. It must not
attribute the already-observed square inference gain to new training.

`cross_policy.csv` evaluates:

| Selected weights | Inference aspect_384 | Inference square_256 |
| --- | --- | --- |
| Trained aspect_384 | Own-policy result | Cross-policy result |
| Trained square_256 | Cross-policy result | Own-policy result |

Both mask and global pooling are measured. Own-policy results are reused from
the normal selected-checkpoint evaluation; the two additional cross-policy runs
extract both pooling modes together. Compare weights under the same inference
policy to assess training effects. This compares selected checkpoints, not
necessarily the last epochs. If both select epoch zero, the cross-policy grid
will mostly reproduce the inference-only diagnostic.

Both selected checkpoints also run the same baseline tracker on the full
conventional KITTI validation sequences to check preservation. A crop inference
gain does not automatically improve full-frame tracking. No VRAI final test is
evaluated; calibration has informed earlier choices and this experiment, so these
are development results, not a new blind test. Geometry and resolution change
together here; earlier crop diagnostics separately measured each factor.

## Outputs

- `checkpoints/crop_training/{aspect_384,square_256}.{last,best}.pt`
- `metrics/crop_training/{aspect_384,square_256}/training.csv`
- `metrics/crop_training/cross_policy.csv` and weight/manifest provenance YAML
- `metrics/crop_training/summary.html` and `summary.csv`

Existing experiment checkpoints and preprocessing settings are retained. Recovery
rejects policy/configuration changes. Training updates are deterministic by
schedule/augmentation seed across resumes; GPU operations need not be bitwise
deterministic. Skipped AMP updates are reported and can break equal successful
update counts. Interrupted evaluation can reuse complete KITTI sequences;
retrieval/cross-policy inference is rerun.
