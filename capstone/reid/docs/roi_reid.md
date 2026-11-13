# Dedicated spatial ReID adapter on shared encoder ROIs

## Correction to the earlier architecture explanation

Upstream `Co-DETR/projects/models/co_detr.py::_mask_forward` assigns
`mask_feats=ins_feats`, where `ins_feats` comes directly from `mask_roi_extractor`
applied to the shared encoded feature pyramid. It is **not** the output of the
RefineMask convolution/fusion stages. The existing ReID head already uses raw
encoder ROI features. Merely extracting those ROIs a second time would not
change its representation.

This experiment instead asks whether **dedicated spatial processing before
pooling** helps ReID. It adds trainable capacity; it does not test encoder
features versus segmentation-decoder features.

## Architecture

```text
One ViT-L backbone + shared encoded feature pyramid
  ├── Co-DINO detection heads → vehicle boxes
  └── ROIAlign → raw object-region features (256 × 14 × 14)
        ├── RefineMask → predicted soft instance mask
        └── ReID input:
              raw_roi:      raw ROI features
              reid_adapter: raw + Conv3×3 → GroupNorm → GELU → Conv3×3
                    ↓
              predicted-mask pooling
                    ↓
              existing identity projection → normalized 256D embedding
```

The adapter is a separate identity-specific processing branch. It reuses the
same ROI tensor rather than duplicating ROIAlign computation. Both mask and
identity branches receive gradients from the identity objective through soft
mask-guided pooling, just as before. The adapter is also used for global pooling
at evaluation and for full-frame detection embeddings at inference.

The last convolution starts at zero, so the residual initially contributes
nothing: **both variants produce identical initial embeddings**. The existing
classifier and projection weights are loaded unchanged. New parameters are only
under `reid_spatial_adapter.*`; unexpected warm-start mismatches are rejected.
The adapter adds **1,180,160 parameters** and two spatial convolutions per ROI.
It is part of inference and is not claimed to be faster than the control.
Older checkpoints/configurations keep the original architecture.

## Run

From `capstone/reid`, with the PC virtual environment active:

```bash
python train.py compare-roi
```

For SSH:

```bash
tmux new -s roi-reid
python train.py compare-roi
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t roi-reid`.
Repeat the same Python command to resume. It checkpoints every 100 cycles and
at epoch boundaries; each cycle contains two optimizer updates. Do not run two
writers against these output paths. Completed arms are skipped.

A bounded GPU check, without trained checkpoint writes:

```bash
python train.py compare-roi-check
```

Reevaluate completed experiments:

```bash
python benchmark.py compare-roi
```

## Matched comparison

`configs/roi_reid.yaml` defines **raw_roi** and **reid_adapter**. Both start from
the pinned, latest distilled joint best checkpoint. Both use joint KITTI
box/mask/identity supervision plus VRAI identity classification/triplet losses.
Neither uses new distillation. They share four epochs of 1,000 cycles (8,000
scheduled updates per arm), identical identity data, sampled KITTI pairs, VRAI
crops, augmentation seeds, optimizer schedules and the final four trainable
backbone blocks. Only the additional spatial adapter differs. It is optimized
with the head learning rate. Parameters/compute are not matched because adding
this branch is the intervention.

The reusable trainer lives in `utils/experiments/reid_balance/training.py`;
experiment paths/configuration and root orchestration live in
`utils/experiments/roi_reid/`. The adapter lives in
`utils/teacher/reid_branch.py`. All crop, KITTI and full-frame identity paths use
`CoDINOJoint._roi_identity`, so they consistently apply the branch.

Best checkpoint selection uses full VRAI calibration mAP with an epoch-zero
fallback. Both selected checkpoints are evaluated with mask and global pooling
on calibration, and with the same baseline tracker on conventional KITTI
validation. Final VRAI test is not evaluated. Results are development evidence;
one seed is preliminary and the baseline was already exposed to earlier data
and distillation training.

Outputs are independent of previous runs:

- `checkpoints/roi_reid/{raw_roi,reid_adapter}.{last,best}.pt`
- `metrics/roi_reid/{raw_roi,reid_adapter}/training.csv`
- `metrics/roi_reid/summary.html` and `summary.csv`

The existing no-distillation and matched-loss comparison checkpoints are not
used as destinations.
