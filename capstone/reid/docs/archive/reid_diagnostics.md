# Diagnose the joint ReID gap

Run from `capstone/reid`, with the PC virtual environment active:

```bash
python benchmark.py reid-diagnostics
python train.py reid-overfit
```

Both commands launch the existing isolated teacher environment. Settings are in
`configs/reid_diagnostics.yaml`. Neither saves trained production weights. The
network overfit check restarts from the configured checkpoint on every run;
it is a bounded diagnostic, not a resumable training experiment.

## Pooling comparison

- A seeded subset of **64 VRAI calibration identities**, all their query/gallery
  crops, compares global and predicted soft-mask pooling. Features and weights
  are identical between variants. This smaller gallery makes these scores
  easier than the complete final test; do not compare their values directly.
- **48 KITTI validation frames** sampled after temporal subsampling compare
  global, predicted-mask and oracle GT-mask pooling. Every variant uses the same
  **GT boxes**, shared encoded features, mask-branch ROI features and ReID head.
  It tests pooling independently of detection failures. Identities are scoped
  to sequences; each repeated identity's earliest sampled frame is its query,
  later frames form the gallery. This is temporal retrieval, not cross-camera
  matching. IDs observed only once are omitted.
- Mean ROI mask IoU uses threshold 0.5, equally weighted per instance. It is not
  full-image IoU or mask AP. VRAI has no GT masks in the current data, so oracle
  aerial pooling cannot be measured here.

## Two levels of overfit checks

1. **Cached-feature head:** 16 VRAI training identities, two images per camera
   where available; 500 full-batch steps. Copies the projection and fits a fresh
   16-way normalized classifier with cross-entropy. Frozen ROI features are
   extracted once, without augmentation. Compare global versus predicted-mask
   inputs. Classification memorization does not imply perfect retrieval;
   triplet/teacher losses are intentionally absent from this isolated check.
2. **Real network path:** four VRAI training identities, one image from each of
   two cameras, fixed eight-image batch. 50 gradient steps through the actual
   predicted-mask crop embedding path, updating its ROI/mask/ReID components and
   final four backbone blocks. Uses original identity classification and triplet
   losses with the existing full identity classifier. Finite nonzero backbone
   and mask-head gradients are checked after successful AMP updates. This
   intentionally removes detector/mask objectives and distillation so those
   competing losses cannot prevent memorization. It does not assess preservation
   of detection/masks. A four-identity gallery is easy and may start at 100%
   Rank-1; classification loss and accuracy remain useful trainability checks.

All memorization images are from training identities. Retrieval reported during
these checks reuses the fitted images: **it is not a held-out score**. No VRAI
final embeddings are extracted by these commands. The existing data loader
checks disjoint train/calibration/final IDs. KITTI validation remains diagnostic
validation, not a new blind test.

## Outputs

`metrics/reid_diagnostics/summary.html`, `pooling.csv`, `overfit.csv` and
`provenance.yaml` describe pooling and head fitting. The second command writes
`network_overfit.csv`, `network_overfit.html`, `network_overfit.yaml`. Provenance
records source weight hash and sampled IDs/frames or paths. No feature caches,
images, NPZs or model checkpoints are written.

If masks help and the tiny network fits, prioritize controlled experiments on
identity representation and multi-task loss balance. If oracle masks help much
more, improve target-domain mask supervision. If the network cannot fit, inspect
its gradients, preprocessing and optimization before another full training run.
These tests narrow hypotheses; they do not alone prove a cause for the specialist
versus joint-model gap. The matched continued-training control is still needed
to attribute gains to distillation.
