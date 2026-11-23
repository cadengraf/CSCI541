# Frozen specialist comparison on VRU

From `capstone/reid`:

```bash
../.venv/bin/python benchmark.py vehicle-vru
```

This compares the original VRAI-adapted TransReID specialist, the final epoch-3 degradation encoder, and the VeRi-pretrained camera-agnostic reference. It performs no training or threshold selection. Device selection is always automatic.

Configuration: `configs/vehicle_vru_benchmark.yaml`. Results: `metrics/vehicle_vru_benchmark/summary.md`, `summary.csv`, and `report.yaml`. Working extraction checkpoints are under `data/vehicle_vru_benchmark/`; rerun the same command to resume. Long runs can be started inside an existing tmux session. Outputs and checkpoints are separate from the older `benchmark.py vru` joint-model comparison.

The full existing split contains 83,595 queries and 8,000 gallery identities, with one fixed gallery crop per identity (local seed 42). VRU training IDs and image paths must be disjoint from evaluation; query/gallery images must be disjoint. Camera labels are absent in these manifests, so this is not a camera-filtered cross-camera protocol. Single-positive mAP equals reciprocal rank and must not be numerically compared with multi-positive VRAI mAP. Results evaluate crop retrieval, not detection, masks, or video return recovery. Different pretrained/adaptation conditions remain part of each model condition.

A real-input 16-identity smoke check and completed-checkpoint resume were verified separately under `check16/`. Those scores are not full benchmark results. No full VRU training or full VRU evaluation was started while preparing the command.
