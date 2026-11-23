# RF-DETR Seg: change discovery, keep ReID and tracking fixed

From `capstone/reid`, with `.venv` active:

```bash
tmux new -s detector-compare
python -u benchmark.py detectors
```

This is a **full user-launched benchmark**, not training. It compares your AdaSP joint checkpoint's Co-DINO/RefineMask detector against pretrained **RF-DETR Seg Large**, using the same original TransReID selection, 256-square crop transform and local Hungarian tracker. Both evaluate cars on all nine KITTI MOTS validation sequences, with score threshold 0.3 and mask threshold 0.5. No tracker tuning or global recovery changes.

Outputs live in `metrics/detector_compare/codino_vs_rfdetr_seg_large_v1/`: component CSV/YAML metrics, official RLE predictions, two annotated videos and logs. The existing `milestones.html` gains a detector-comparison table only when a first result/new best is recorded. No additional HTML report is retained. No ReID training or retrieval accuracy improvement is implied by replacing discovery.

## Resume and timing

Repeat the same command after interruption. Each detector has its own process/environment and checkpoint per sequence every 25 frames. The tracker, recovery audit, output offset and timing are saved; uncommitted RLE tails are truncated on resume. Completed model results are verified/reused. A changed config, source, checkpoint, dataset or recorded environment requires a new `run` in `configs/detector_compare.yaml`.

The validation latency includes decode, detector forward, mask filtering, specialist, association, RLE and checkpoint writes. GT auditing and TrackEval are offline and excluded. A separate **fresh 200-frame sequence-0010 video pass** measures decode through annotation and video encoding/finalization, excluding initialization. This timing does not use cached detections or embeddings. Both models are warmed up first. Native PyTorch inference is used; no TensorRT/compilation speed claim is made.

Two-frame integration check only:

```bash
python benchmark.py detectors --check
```

The check produces no validation scores or scorecard updates. RF-DETR dependencies are isolated under `capstone/.rfdetr-env`, sharing the existing large Torch installation through a read-only path; the modern and legacy environments are preserved. Package 1.12.1 and the upstream checkpoint MD5 are pinned in YAML. Setup/downloads are automatic if missing. Logs appear in tmux and `logs/`.

## Interpret the comparison correctly

Co-DINO uses KITTI/VRAI-finetuned weights, aspect-preserving full-frame resizing and an AMP backbone/FP32 heads; its current forward also computes an unused joint embedding. RF-DETR uses COCO-pretrained weights at its native 504-square resolution and its own runtime. These are practical replacement configurations with different exposure, resolution and precision, not a matched-training architecture ablation.

Inspect HOTA/DetA/AssA, IDF1, sMOTSA, switches, local visibility-gap recoveries and false reconnects beside both latency scopes. A detector can change crop geometry and therefore specialist performance despite unchanged specialist weights. Cross-camera/long-term ReID is not established by this test. After inspecting these results, consider matched vehicle-mask fine-tuning, then the shared RF-DETR ReID adapter; Cutie/McByte++ remain later controlled association changes.

Official API/weights: [RF-DETR segmentation](https://rfdetr.roboflow.com/latest/learn/run/segmentation/), [weight registry](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/assets/model_weights.py).
