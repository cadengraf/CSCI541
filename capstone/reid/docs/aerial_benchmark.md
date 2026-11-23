# Full aerial retrieval benchmark

From `capstone/reid`, run or resume all four completed mixed-domain models:

```bash
../.venv/bin/python benchmark.py aerial
```

The launched GPU queue runs in tmux with a live log window:

```bash
tmux attach -t aerial_benchmark
```

Configuration: `configs/aerial_benchmark.yaml`, with `device: auto`. The queue evaluates each model on both datasets and advances automatically. It freezes checkpoint checksums, development-selected epochs, training provenance, and test manifest checksums in `metrics/aerial_benchmark/selection.yaml` before extracting test features. Test performance never selects a checkpoint or fits a rejection threshold. No deployment default is changed.

VRAI uses the existing local test split: 1,223 queries, 2,199 gallery images, and 630 identities with disjoint query/gallery cameras. This is a split derived from the available VRAI training collection, not the official VRAI test leaderboard. VRU uses the existing fixed single-gallery protocol: 83,595 queries and 8,000 gallery identities. Its manifests lack camera labels, so its result does not establish cross-camera accuracy. With one positive per query, VRU mAP is mean reciprocal rank.

Do not compare the VRU development mAP directly to this full benchmark: development selection has only 354 gallery identities. Historical comparisons also need matching gallery manifests and training conditions. The archived VRU-trained MSINet reference used different training conditions.

Each model reports Rank-1 and mAP separately by dataset. Open-set diagnostics withhold every fifth sorted gallery identity, use one reference per remaining identity, and apply that model's previously fitted source-specific score/margin thresholds. Correct known acceptance, wrong known acceptance, and unknown false acceptance are separate results. These crop retrieval diagnostics do not measure disappearance recovery in video.

Results appear under `metrics/aerial_benchmark/`: `queue.yaml`, per-model YAML, `summary.csv`, and `summary.md`. Necessary resumable feature shards live under `data/aerial_benchmark/`; no image or video copies are produced. Extraction checkpoints every 64 batches. Rerunning the same command reuses committed shards and completed results. Changed inputs, checkpoints, or implementation require new state/output paths. The queue lock rejects simultaneous duplicate runs.

`benchmark.py aerial --check` evaluates only 16 identities per dataset in separate `check16` directories. It verifies integration and must never be reported as full benchmark accuracy. Two CPU tests cover exact rejection-protocol agreement and interrupted/resumed extraction; the GPU check loads and evaluates all four real checkpoints on real images.

Timing covers this retrieval extraction/scoring job, not the online detector, masks, association, or rendering pipeline. Online speed and video identity metrics require separate evaluation.
