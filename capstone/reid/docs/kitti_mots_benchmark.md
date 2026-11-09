# Clean KITTI MOTS vehicle benchmark

Run from `capstone/reid` with the project's virtual environment active:

```bash
python train.py kitti-mots
```

This trains a new checkpoint and automatically runs tracking and TrackEval
after training finishes. Settings live in
[`configs/kitti_mots.yaml`](../configs/kitti_mots.yaml). The existing
`config.yaml` experiments and their checkpoints remain development runs.

For an SSH session, start `tmux new -s kitti-mots` first. Detach with Ctrl+B,
then D. Training saves every 250 steps and at each epoch boundary. Run the
same command to resume after an interruption.

## Protocol and initialization

| Partition | Sequences |
| --- | --- |
| Training | 0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20 |
| Validation | 2, 6, 7, 8, 10, 13, 14, 16, 18 |

This is the conventional 12/9 partition used in KITTI MOTS research.
The network initializes from TorchVision's COCO-trained Mask R-CNN
ResNet-50-FPN v2 weights; it does not load the old KITTI-only or KITTI+VRAI
checkpoint. It trains box, mask, identity-classification, and triplet losses
on the KITTI training partition. VRAI is excluded from this first clean run
so that extra identity supervision can be studied as a separate experiment.
The new checkpoint is `checkpoints/kitti_mots/joint.pt`.

The benchmark rejects checkpoints whose saved split/protocol differs or
whose training is incomplete. Training also checks saved sequence lists
before resuming. Previously trained checkpoints cannot be turned into clean
checkpoints by merely changing the validation list.

## Tracking and mask export

For every frame, the network supplies vehicle boxes, instance masks, and
128D identity embeddings. A simple Hungarian tracker combines predicted
motion, box IoU, and cosine appearance similarity. Nearby detections can
recover tracks after a short absence. This tracker is a project baseline;
it does not implement ByteTrack, MOTSNet's exact association, or long-term
cross-camera handoff.

Masks are thresholded, and overlapping pixels are assigned to the detection
with higher confidence. Tiny remaining masks are discarded. These thresholds
are in the config; ground truth never enters prediction or association.
Each sequence resets its tracker. Every validation frame is processed,
including frames without annotated cars. Masks/IDs are exported in KITTI
MOTS RLE text format under `data/kitti_mots/benchmark/joint/data/`.

Only completed sequence exports with matching checkpoint and configuration
provenance are reused after interruption. If weights or tracker settings
change, select a new `tracking.name` to preserve the previous experiment.

## Metrics and reports

The pinned official TrackEval implementation evaluates mask overlap, car
identities, and ignore regions. It reports HOTA, DetA, AssA, IDF1, identity
switches, and CLEAR metrics. With mask similarity, TrackEval's `sMOTA`,
`MOTA`, and `MOTP` are labeled `sMOTSA`, `MOTSA`, and `MOTSP` in our report.
All score columns are percentages; identity switches/TP/FP/FN are counts.
The adapter temporarily supplies removed NumPy dtype aliases to the original
TrackEval code. It does not replace metric formulas.

Open `metrics/kitti_mots/summary.html` after completion. The same scores are
saved in `summary.csv`, official per-sequence details in each tracker's
subfolder, and end-to-end timing in `timing.csv`. Timing uses batch size one
and includes PNG decoding, inference, output transfer, association, and mask
encoding/export. Compare speed only on the same hardware and with the same
input/preprocessing. Cached timings from different devices are kept separate.

To run or repeat evaluation independently:

```bash
python benchmark.py kitti-mots
```

To rerun TrackEval on existing exports without network inference:

```bash
python benchmark.py kitti-mots-score
```

The default report contains only models actually evaluated. Historical
published milestones are TrackR-CNN's 76.2 car sMOTSA and MOTSNet's 78.1 car
sMOTSA. They are reference targets, not reproduced baselines. Their training
data/pretraining and architecture differ. Reproduce or import their results
on this same validation partition before claiming a controlled improvement.
Box AP50, matched-mask IoU, and VRAI retrieval mAP are different metrics.

## Adding an external baseline

Place the baseline's prediction files in a separate folder under `data/`.
Each validation sequence needs one `0002.txt`, `0006.txt`, etc. The files must
contain non-overlapping masks and persistent IDs in KITTI MOTS RLE format.
Do not enter published scores into the evaluator as if measured locally.

Example config entry (the folder must exist and contain actual predictions):

```yaml
external_trackers:
  MOTSNet:
    predictions: data/kitti_mots/baselines/motsnet/data
    protocol:
      protocol: kitti_mots_standard_val_v1
      train_sequences: [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
      val_sequences: [2, 6, 7, 8, 10, 13, 14, 16, 18]
```

Put that mapping under `tracking.external_trackers`, then run
`python benchmark.py kitti-mots-score`. External training provenance is
declared by the user; prediction files alone cannot prove which images a
baseline trained on. Record its weights, code revision, extra training data,
and inference settings alongside the experiment.

## Installation and verification

The benchmark dependencies are already installed in this project's venv.
For a new environment:

```bash
python -m pip install -r requirements-mots.txt
python benchmark.py kitti-mots-check
python -m unittest utils.full_frame.tests.test_mots -v
```

The check command exercises a real KITTI loss batch, three-frame inference,
mask/ID export, and TrackEval in a temporary directory. It uses untrained
weights for inference; its smoke scores are not research results. The tests
check identity continuity, overlap removal, contaminated-checkpoint rejection,
perfect tracking, a known identity switch, and ignored-region detections.

Sources: [TrackEval](https://github.com/JonathonLuiten/TrackEval),
[conventional split in PointTrack](https://github.com/detectRecog/PointTrack/blob/master/datasets/KittiMOTSDataset.py),
[MOTSNet Table 2](https://arxiv.org/pdf/1912.02096),
[TrackR-CNN Table 2](https://arxiv.org/pdf/1902.03604).
