# Vehicle ReID

Open [milestones.html](milestones.html) for local results, previous/best comparisons and published reference tables. One HTML is updated only when a comparable best improves; unsuccessful runs print their comparison and retain a small metric record.

## Run

From `capstone/reid`, with `.venv` active:

```bash
tmux new -s reid-bench
python -u benchmark.py suite
```

Use `configs/benchmark_suite.yaml` for the run name, selected joint checkpoint and enabled evaluations. Give a changed model a new run name; repeat the same command to resume. [Run guide](docs/benchmark_suite.md).

The user launches all long runs. Calling a root script without a command prints help only.

Completed metric test: AdaSP reached 33.62% held-out joint mAP versus 27.66% for the matched triplet control. [Recipe and commands](docs/joint_metric.md).

Completed detector baseline: RF-DETR was faster but less reliable than Co-DINO. [Comparison protocol](docs/detector_compare.md).

Next adaptation: `python train.py rfdetr`, then `python benchmark.py rfdetr`. [Data split, checkpoints and recipe](docs/rfdetr_finetune.md).

## Layout

- `train.py`, `benchmark.py`, `demo.py`, `prepare.py`: the four entry points.
- `configs/`: YAML experiment settings and published reference values.
- `utils/`: implementation modules; [map](utils/README.md).
- `docs/`: [architecture](docs/architecture.md), [roadmap](docs/roadmap.md), [guides](docs/README.md) and archived experiments.
- `requirements.txt`: modern-environment installation entry point; specialized lists live in `requirements/`. The isolated joint-model environment retains its own dependencies under `utils/teacher/`.
- `data/`, `checkpoints/`, `metrics/`: local data, trained states and generated evidence; ignored for new Git additions. Completed suite scratch caches are removed by default. Previously saved training checkpoints and datasets are preserved.

The joint model shares ViT-L / Co-DINO / RefineMask features with a dedicated 256D ReID branch. Independent TransReID remains an accuracy reference and optional training teacher. Retrieval, recovery, cross-camera tracking and inference cost are evaluated separately.
