# Specialist vehicle ReID teacher

## Measured starting point

Three existing, vehicle-trained candidates were compared on **all VRAI
calibration identities**. The winner was chosen by calibration mAP before running
the final local retrieval evaluation.

| Candidate | Calibration Rank-1 | Calibration mAP | Final Rank-1 | Final mAP |
| --- | ---: | ---: | ---: | ---: |
| Compact warm-start ReID | 59.28% | 56.52% | 58.30% | 55.76% |
| **MSINet, released clean-VRU checkpoint** | **77.61%** | **75.65%** | **77.27%** | **75.11%** |
| TransReID, VeRi pretrained, camera-agnostic adaptation | 43.49% | 40.47% | 41.21% | 38.86% |

MSINet is the selected reference teacher. A bigger architecture is not selected
automatically. Its existing vehicle-trained checkpoint is used directly; it has
not been fine-tuned on VRAI here. The compact reference reproduces the previous
58.30% / 55.76% final result under this evaluator.

Results are [calibration.html](../metrics/reid_teacher/calibration.html) and
[final.html](../metrics/reid_teacher/final.html), with CSVs alongside them. The
selected model's path, SHA-256 and calibration score are recorded in
`checkpoints/reid_teacher/selected.yaml`. The source revisions and pretrained
hashes are pinned in `configs/reid_teacher.yaml`.

These are **local cross-camera splits of VRAI training images**, not published
official VRAI test scores. Calibration has 1,228 queries / 2,179 gallery images;
final retrieval has 1,223 queries / 2,199 gallery images. Training uses 52,966
images / 5,042 vehicle IDs. Train, calibration and final identity/path separation,
and query/gallery camera separation, are checked before running. This local final
set has already been reported in earlier project experiments; it is not a new
blind test set. Do not use its results to choose hyperparameters or epochs.

All models use their appropriate image preprocessing and cosine similarity with
same-camera gallery images excluded. Their pretraining datasets and budgets
differ. These are specialist crop-retrieval results, not full-frame detection or
tracking scores. A model chosen here is not automatically installed in the joint
network or tracker; its knowledge can supervise a subsequent distillation step.

## Commands

From `capstone/reid`, with the PC `.venv` active:

```bash
python benchmark.py reid-teacher
```

This compares candidates on calibration only and writes a provisional selection.
It does not evaluate final queries or change the frozen final selection.

```bash
python train.py reid-teacher
```

This fine-tunes the **vehicle-pretrained TransReID** challenger on VRAI. It is a
long GPU run, not another evaluation of the existing MSINet weights. Its purpose
is to test whether adapting a larger specialist can improve the current teacher.

For SSH:

```bash
tmux new -s reid-specialist
python train.py reid-teacher
```

Detach with **Ctrl+B**, then **D**. Reattach with
`tmux attach -t reid-specialist`. Repeat the training command to resume from the
last completed save. No training beyond short optimizer checks was launched
during implementation.

The run saves a resumable checkpoint every 200 steps and after each epoch.
Model, optimizer, AMP scaler, epoch/step, sampling and RNG states are saved with
atomic replacement. Recovery rejects changed settings, manifests or identity
mapping. Work since the last save can be lost on interruption. Run one writer per
checkpoint path.

Other commands:

```bash
python train.py reid-teacher-check
python benchmark.py reid-teacher-final
python benchmark.py reid-teacher-setup
```

The check performs real classifier-warmup and full-backbone updates and verifies
finite retrieval embeddings without writing training checkpoints. Setup fetches
the pinned source and authors' checkpoint and verifies its SHA-256; it expects
the two existing reference checkpoints and prepared VRAI manifests. The model
runs in the normal PC environment; the legacy detection teacher environment is
not used. Requirements are in `utils/reid_teacher/requirements.txt`.

The final command selects by calibration mAP and reports the fixed candidates
on final query/gallery. It can finalize pretrained candidates before fine-tuning
starts. Once a fine-tuning run exists, it requires that run to finish before final
evaluation. Training invokes this comparison automatically on completion.

## TransReID fine-tuning recipe

- Official ViT-B TransReID with JPM local feature branches and stride 12; direct
  256×256 crop resize, mean/std 0.5, concatenated 3,840D retrieval embedding.
- Initialize the backbone, global/local branches and normalization layers from
  the authors' **VeRi-trained** checkpoint, not just ImageNet features.
- Replace the five identity classifiers because VRAI uses different IDs.
- Disable VeRi camera/view side embeddings: VRAI's camera IDs and unavailable
  viewpoint labels cannot be mapped to VeRi's learned camera/view table. This is
  a transfer adaptation, not a reproduction of published VeRi scores.
- Batch: eight identities × four images, with two distinct cameras per identity
  and two samples from each camera. Sparse cameras may sample with replacement.
- Average identity cross-entropy and normalized batch-hard triplet losses across
  the global and four local branches. Label smoothing 0.1, triplet margin 0.3.
- Thirty epochs, 1,600 sampled updates each; one classifier warmup epoch, then
  full backbone/branch fine-tuning. AdamW, smaller shared-network learning rate,
  cosine decay, clipping, mixed precision and activation checkpointing.
- Horizontal flips, padded random crops, modest color jitter and random erasing
  are applied during training only. No augmented image files are saved.
- Preserve the unadapted pretrained model as epoch-zero best. Select later
  checkpoints by **full calibration mAP**, never final mAP. Fine-tuning is allowed
  to lose to the pretrained model or MSINet.

Sampling is identity balanced; 32 × 1,600 draws per epoch do not guarantee visiting
each training image. All schedule and path settings are in the dedicated config.
No mask loss, detector loss or KITTI training is involved in this specialist run.

## Loading the selected teacher

`utils/reid_teacher/evaluation.py:load_selected()` returns a frozen model, its
required PIL preprocessing and selection metadata. Its output is L2-normalized
identity embeddings. It validates the selected weight hash and calibration
manifests before loading. This provides a stable interface for later distillation;
the distillation loss and student training are not implemented in this step.

## Code map

| File under `utils/reid_teacher/` | Responsibility |
| --- | --- |
| `data.py` | Config, split audit, camera-balanced sampling, augmentation |
| `model.py` | Official model adapter, camera-agnostic transfer, identity losses |
| `training.py` | Fine-tuning, calibration selection, optimizer and resume |
| `evaluation.py` | Matched retrieval, candidate comparison, frozen teacher loader |
| `setup.py`, `requirements.txt` | Source, vehicle pretrained weights and dependencies |

Sources: [TransReID](https://github.com/damo-cv/TransReID),
[MSINet](https://github.com/vimar-gu/MSINet),
[released UAV vehicle benchmark/checkpoints](https://github.com/tranminhvu945/Benchmarking-ReID).
