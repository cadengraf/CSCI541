# Implementation map

Run commands from the parent directory. The four root scripts are the user
interface; files here implement their work.

| Folder | Responsibility | Start reading |
| --- | --- | --- |
| `core/` | Network, datasets, training, evaluation, metrics, checkpoints, target tracking | `model.py`, then `training.py` |
| `data_sources/` | Download or convert datasets into identity and mask manifests | `visdrone.py`, `vrai.py`, `prepare_aerial.py` |
| `visualization/` | Render VisDrone video and select a target | `video.py`, `target_search.py` |
| `crossdrone/` | BuckTales preparation, retrieval, and video | `data.py`, `retrieval.py`, `video.py` |
| `benchmarking/` | Load comparison models, measure them, create charts | `benchmark.py`, `models.py`, `report.py` |
| `experiments/` | Matched ReID-only versus joint training | `compare.py` |
| `UAVidToolKit/` | Third-party UAVid tooling | Its own `README.md` |

The model accepts a **crop of one object**. `core/model.py` returns a semantic
mask and a 128D identity embedding. `core/training.py` trains both heads using
the identity sources and VOC masks selected in `config.yaml`. The video code
uses dataset boxes to supply crops; `core/tracking.py` maintains the selected
target between frames.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the model diagram and command
outputs.
