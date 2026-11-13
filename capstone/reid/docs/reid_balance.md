# Matched identity-only versus joint continuation

This experiment asks whether the detection/segmentation objectives hurt the
identity representation in the **existing** Co-DINO/RefineMask network. It does
not replace the backbone and does not reproduce a standalone TransReID model.

## Run

From `capstone/reid`, with the PC virtual environment active:

```bash
python train.py compare-reid
```

For SSH:

```bash
tmux new -s reid-balance
python train.py compare-reid
```

Detach using Ctrl+B, then D. Reattach with `tmux attach -t reid-balance`.
The command trains `reid_only`, then `joint`, in separate GPU processes, then
evaluates both and creates `metrics/reid_balance/summary.html`. The long run is
not started by setup/checks. Do not launch two copies writing these same paths.

Repeat the same command to resume the current arm. Checkpoints are saved every
100 complete cycles and at epoch boundaries. Each cycle has two optimizer
updates, so interruption can lose up to 100 cycles. Already completed arms are
skipped, and completed evaluation sequences are reused when provenance matches.
The smoke check performs one crowded KITTI pair and one VRAI batch in each arm:

```bash
python train.py compare-reid-check
```

To reevaluate completed checkpoints and rebuild the comparison:

```bash
python benchmark.py compare-reid
```

All options live in `configs/reid_balance.yaml`. Changing the experiment after
it has started causes recovery validation to reject the old checkpoint; use new
checkpoint/output directories for a different experiment.

## What is matched

| Item | Both arms |
| --- | --- |
| Origin | Pinned latest distilled joint best checkpoint, epoch 8 |
| Backbone/head architecture | Same ViT-L, Co-DINO, RefineMask and 256D identity head |
| KITTI identity data | Same consecutive frame pairs and identity labels |
| VRAI identity data | Same 5,042 training IDs, two camera views per sampled ID |
| Batch | One KITTI pair, then four VRAI IDs × two cameras = eight crops |
| Budget | Four epochs × 1,000 cycles = 8,000 scheduled optimizer updates per arm |
| Trainable layers | Last four backbone blocks and existing heads; paths without loss gradients naturally do not update |
| Optimizer | AdamW, head LR 1e-5, backbone LR 1e-6, same cosine schedule and gradient clipping |
| Augmentation | Same image sampling and per-stage torch seeds |
| Selection | Best full VRAI calibration mAP, including epoch-zero fallback |
| New distillation | Disabled in both; both start from weights that previously received distillation |

Each epoch uses a seeded sample of 1,000 distinct training pairs from the 4,050
available pairs. The arms get exactly the same sampled pairs. Pairs can repeat
across epochs. Each stage resets a scoped torch RNG using `(seed, epoch, cycle,
stage)`; VRAI sampling uses a separate local RNG with the same deterministic seed.
Recovery skips completed cycles without changing future image/augmentation
choices. AMP state and optimizer state are restored. Logged batch schedule keys
must agree before the comparison report is generated. Skipped AMP updates are
counted: different skipped counts invalidate a claim of equal successful update
budgets and are flagged in the report.

## The one changed factor

- **reid_only:** KITTI identity classification + triplet losses, followed by
  VRAI identity classification + triplet losses. Detection and mask losses are
  computed by the shared KITTI forward but excluded from backward.
- **joint:** Same identity updates, plus all existing KITTI detection,
  segmentation and mask-quality auxiliary losses in the KITTI update.

The ReID-only arm **still predicts and uses masks**, and gradients from identity
loss can change its mask branch. It is an identity-only training objective, not
an architecture without masks. Allowing masks to change without direct mask
supervision is part of what this experiment measures. Neither arm gets more
identity batches. Forward budgets are similar; backward FLOPs and wall time are
not identical because auxiliary losses contribute gradients only in joint.

## Evaluation and interpretation

Both selected checkpoints are evaluated with global and predicted-mask pooling
on the **same full calibration query/gallery**, plus all conventional KITTI MOTS
validation sequences with the same baseline tracker. The official TrackEval mask
metrics report detection/segmentation/tracking preservation jointly; they are
not independent box AP or mask AP. Individual reports and timing are under
`metrics/reid_balance/{reid_only,joint}/`.

No VRAI final test retrieval is performed. Calibration both selects checkpoints
and compares methods, so these are development scores, not blind test results.
The old final split has already been inspected repeatedly and should not be
represented as a new untouched test for future architecture choices.

- Identity-only improving more is evidence that the auxiliary objectives or
  their weighting constrain ReID **under this schedule**.
- Both staying weak suggests testing the identity representation/path,
  preprocessing and adaptation capacity next.
- Identity-only degrading KITTI masks/detections reveals the preservation cost.

One seed and a shortened continuation budget provide preliminary evidence, not
a definitive explanation of the specialist/joint performance gap. The original
pre-distillation continuation control remains a different experiment; this
comparison does not attribute the earlier improvement specifically to KD.
