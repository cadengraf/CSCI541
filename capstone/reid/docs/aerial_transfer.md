# Unseen aerial dataset evaluation

Run or resume from `capstone/reid`:

```bash
../.venv/bin/python benchmark.py aerial-transfer
```

The launched queue and live log window use:

```bash
tmux attach -t aerial_transfer
```

Settings live in `configs/aerial_transfer.yaml`; the device is automatic. The original VRAI-only TransReID and all four mixed VRAI/VRU TransReID models are frozen before evaluating VisDrone2019-MOT validation vehicles. Their actual fine-tuning manifests exclude VisDrone. Historical compact/joint models in this repository did use VisDrone; those models are deliberately excluded. These specialist checkpoints descend from VeRi-pretrained TransReID weights. We cannot independently audit every image used by upstream pretraining.

This is a local test of transfer to an unseen dataset, using temporal identities within each video. It is not an official VisDrone ReID protocol, a cross-camera benchmark, or a full tracking evaluation. Ground-truth boxes isolate appearance quality from detection and mask errors. No target images enter training, checkpoint selection, or rejection fitting. Previously reported VRU scores are not directly comparable to this much smaller, sequence-specific gallery.

Cars, vans, trucks and buses are included (VisDrone category IDs 4, 5, 6, 9); target ID zero is valid. Ignore regions and score-zero entries are excluded. Each sequence has a fixed midpoint: reference enrollment ends 30 frames before the midpoint and queries begin 30 frames after it. Each enrolled identity contributes its first eligible crop, with minimum side 16 pixels, no truncation, and at most partial occlusion. Queries require minimum side 12 pixels, can be occluded/truncated, and are sampled chronologically at least 30 frames apart, up to eight per identity. All gallery observations precede every query by at least 60 frames. No future crops update the gallery. New identities appearing after enrollment are retained as natural unknowns.

The initial protocol yields 146 references and 622 queries across six vehicle-containing sequences. The pedestrian-only sequence is skipped. Counts include unknown queries; the closed-set retrieval count is smaller. Selected crop pixels are SHA256-compared to the cached original and mixed-model training crops before inference. Exact overlap aborts the run. This does not prove physical-identity separation or detect every near duplicate.

Closed-set Rank-1 and single-positive mAP use all enrolled identities from the same sequence. Results include per-sequence counts, size and occlusion slices. Different videos are never treated as known cross-camera identities or pooled as an artificial common camera.

Open-set scoring additionally withholds every fifth sorted enrolled identity per sequence. Both natural and withheld unknown false acceptance are reported separately. The original model uses its archived 1% empirical-budget VRAI calibration point. Each mixed model is evaluated at both previously fitted VRAI and VRU points; neither is selected using VisDrone. Known correct acceptance, wrong known acceptance and unknown false acceptance are distinct. Empirical source budgets do not guarantee target error rates.

`metrics/aerial_transfer/selection.pt` freezes all checkpoint checksums, thresholds, source annotations, selected observations and crop hashes before scoring; `protocol.yaml` contains the human-readable provenance. `queue.yaml` records automatic sequential progress. Per-model YAML includes metrics and extraction cost; `summary.md` and `summary.csv` compare models. Timing includes image reads/cropping/transforms/embeddings but not detector, mask, tracker, rendering or model-loading costs. It is not end-to-end online FPS.

`review.mp4` shows a bounded, curated error review comparing the original specialist and AdaSP (chosen before this test because it led the preceding VRU evaluation). It shows actual later query crops and earlier open-set retrieved references, with source-fitted decisions. Natural unknowns, disagreements, occlusions and correct/incorrect outcomes are mixed deliberately; this video is not a representative accuracy sample. No image copies or separate feature exports are generated. Necessary resumable extraction state lives under `data/aerial_transfer/`; rerunning reuses committed batches and complete results. A queue lock prevents duplicate runs. Changed frozen inputs require separate output/state paths.

Two CPU tests verify chronological separation, zero-valued track IDs, gallery isolation by video, natural unknowns, synthetic withholding and known retrieval. GPU execution verifies all five real checkpoints against the actual dataset. Small correlated track samples limit conclusions: report counts and per-video variation, and do not use this local protocol to claim leaderboard performance.

Annotation definitions: [official VisDrone MOT toolkit](https://github.com/VisDrone/VisDrone2018-MOT-toolkit#mot-submission-format).
