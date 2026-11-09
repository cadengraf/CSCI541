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
| `full_frame/` | Shared vehicle detector, instance masks, 128D ReID head, KITTI and VRAI training and evaluation | `model.py`, `kitti.py`, `training.py`, `multiview.py` |
| `teacher/` | Pretrained Co-DINO/RefineMask teacher with shared 256D ReID, clean KITTI/VRAI fine-tuning and evaluation | `model.py`, `training.py`, then [guide](../docs/teacher.md) |
| `reid_teacher/` | Specialist vehicle ReID selection, TransReID fine-tuning and frozen teacher loader | `evaluation.py`, `model.py`, `training.py`, then [guide](../docs/reid_teacher.md) |
| `tracking/` | Fixed-teacher tracker comparison, adapted McByte++/EdgeTAM, selected-object CUTIE MP4 | `comparison.py`, `mcbyte.py`, `video.py`, then [guide](../docs/tracking.md) |
| `vendor/Co-DETR/` | Pinned upstream teacher source, fetched during setup and excluded from Git | Its own `README.md` |
| `UAVidToolKit/` | Third-party UAVid tooling | Its own `README.md` |

The original model accepts a **crop of one object**. `core/model.py` returns a semantic
mask and a 128D identity embedding. `core/training.py` trains both heads using
the identity sources and VOC masks selected in `config.yaml`. The video code
uses dataset boxes to supply crops; `core/tracking.py` maintains the selected
target between frames.

The new `full_frame/model.py` takes complete frames. Its default COCO-pretrained
ResNet-50-FPN features feed box, instance-mask, and mask-pooled ReID heads in one forward
pass. `full_frame/kitti.py` loads KITTI MOTS car boxes, masks, and track IDs;
`full_frame/training.py` trains all heads on KITTI. `full_frame/multiview.py`
continues the same network with two-camera VRAI vehicle identities while KITTI
still supervises boxes and masks. `full_frame/evaluate.py` measures local KITTI
validation, and `full_frame/cross_view.py` scores VRAI crop retrieval. The
current full-frame network generates masks for every detected vehicle.

The clean paper-comparison path uses `full_frame/protocol.py` to enforce the
standard split, `full_frame/tracker.py` to associate detections over time,
`full_frame/mots_benchmark.py` to export complete validation sequences, and
`full_frame/mots_evaluation.py` to call the official TrackEval metrics. Its
settings are in `configs/kitti_mots.yaml`; user commands remain `train.py`
and `benchmark.py` in the project root.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the model diagram and command
outputs.
