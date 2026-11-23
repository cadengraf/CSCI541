# Next: adapt RF-DETR's boxes and masks to KITTI

The pretrained front end was faster but less reliable: HOTA 70.23 versus Co-DINO 77.98, 200 versus 46 ID switches, and 57 versus 16 false reconnects. This test fine-tunes **RF-DETR Seg Large detection/segmentation only**. TransReID, local tracking, score thresholds, mask rules, resolution and evaluation sequences stay fixed. No RF-DETR ReID head is trained yet.

## Train

From `capstone/reid`, with `.venv` active:

```bash
tmux new -s rfdetr-train
python -u train.py rfdetr
```

Settings: `configs/rfdetr_finetune.yaml`. The pinned `.rfdetr-env` and training dependencies are prepared on this machine; their optional install list is `requirements/rfdetr-training.txt`.

| Use | Sequences | Frames |
|---|---|---:|
| Gradient updates | 0, 1, 3, 4, 5, 11, 12, 15, 17, 20 | 3,165 |
| Internal development / checkpoint selection | 9, 19 | 1,862 |
| Benchmark only | Official nine validation sequences | 2,981 |

Train 30 epochs at 504-square resolution, batch 2 with four-step accumulation (effective batch 8). Head LR is 1e-4; encoder LR is 1e-5. Native box/classification and mask losses update RF-DETR; EMA and internal-development **mask mAP** select the final checkpoint. The official validation sequences are never exported into the training/development data or used for selection. No automatic test run follows training.

Original RGB images are linked, not copied. Two COCO JSON manifests store compressed instance masks directly; no mask PNGs or augmented image copies are created. Empty car frames remain negative examples. KITTI pedestrians/ignore labels are excluded, consistent with the existing car-only training path.

The COCO car classifier row is retained when adapting to one car class (local ID **0**). Grouped query metadata is repaired before loading: the source has 300 queries × 13 groups; the model retains 200 queries per group. This preserves each group's pretrained rows instead of flat truncation. Inference architecture remains RF-DETR Seg Large with a car-only classifier.

## Resume and benchmark

Repeat `python train.py rfdetr` to resume. Native full checkpoints save every epoch and every 100 optimizer steps; the latest valid optimizer progress is selected. They include model, optimizer/scheduler, scaler when used, RNG and callback state. Native Lightning minibatch ordering after a mid-epoch restart is not promised to be bit-identical. Changed data, recipe or local training implementation requires new output paths. Logs stream to tmux and `metrics/rfdetr_finetune/kitti_v1/training.log`. Detach with Ctrl+B, D; reattach with `tmux attach -t rfdetr-train`.

After training finishes:

```bash
python -u benchmark.py rfdetr
```

This reloads the selected `checkpoint_best_total.pth`, verifies the original benchmark's data, specialist, tracker and Co-DINO weights are unchanged, then repeats full KITTI tracking and fresh annotated-video timing. The original comparison config/results remain intact. Car label 0 is handled explicitly. Results enter the same improvement-only scorecard and latest/previous/best ledger; no extra HTML report is retained.

Training keeps the selected inference checkpoint and final full resume state. Redundant native exports are removed only after successful completion. Existing datasets, pretrained weights and earlier best models are preserved.

Optional bounded integration check: `python train.py rfdetr --check`. It performs one optimizer step/validation batch plus a temporary native checkpoint export/reload, and writes no production checkpoints or benchmark scores. Development numbers printed by this check are not benchmark results.

Inspect HOTA, DetA, mask quality, ID switches/false reconnects and live-video latency. Co-DINO saw all 12 training sequences plus VRAI; RF-DETR uses 10 training and two internal-development sequences. The official validation split was observed in earlier comparisons; it is not a fresh blind test, although its labels remain excluded from gradients and checkpoint selection. Improved results would support this practical adaptation, not a matched-training architecture claim. If useful, test the dedicated shared RF-DETR ReID branch next.

Native training/resume API: [RF-DETR training](https://rfdetr.roboflow.com/latest/learn/train/) and [advanced training](https://rfdetr.roboflow.com/latest/learn/train/advanced/).
