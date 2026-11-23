# Next joint training: change only the VRAI metric loss

The earlier spatial-adapter and identity-only ablations selected epoch zero. This test starts from the later square-crop joint checkpoint and compares **classification + batch-hard triplet** against **the same classification + AdaSP** on VRAI batches. KITTI identity/detection/mask losses remain identical. No architecture, embedding dimension, crop policy, data, augmentation, sampling or association threshold changes.

## Train — user launches

From `capstone/reid`, with `.venv` active:

```bash
tmux new -s joint-metric
python -u train.py joint-metric
```

The command runs control and candidate sequentially in the existing `.teacher-env`. Each receives 8 epochs × 1,500 KITTI-pair updates and the same number of VRAI updates. Four VRAI identities × two views/crops give balanced identity groups for AdaSP; KITTI's uneven groups retain triplet loss. AdaSP temperature is 0.04. Both retain full KITTI supervision, the last four trainable ViT blocks and the existing learning rates. See `configs/joint_metric.yaml`.

Logs stream to tmux and `metrics/joint_metric/{control,candidate}.log`. Repeat the command to resume; completed arms are skipped. Checkpoints save every 100 KITTI/VRAI update pairs and each epoch. Stateless per-step sampling/augmentation matches across arms and resumes. Changed config, manifests or pinned implementation rejects recovery. Avoid simultaneous writers and code edits during the run. Detach with Ctrl+B, D; reattach with `tmux attach -t joint-metric`.

Calibration mAP selects each best checkpoint, including the unchanged epoch-zero fallback. Held-out retrieval/recovery never selects epochs. Source weights remain untouched; outputs go to `checkpoints/joint_metric/{control,candidate}.{best,last}.pt`.

## Benchmark — separate user command

After **both arms finish**:

```bash
python -u benchmark.py joint-metric
```

This invokes the existing suite for each arm: held-out VRAI mask/global retrieval, all KITTI validation, and the unchanged specialist/joint recovery protocols. Expensive frozen specialist sweeps and CityFlow are not repeated. Benchmark configs live under `metrics/joint_metric/`; the baseline suite config is preserved. Results enter the same latest/previous/best ledger and improvement-only `milestones.html`. No additional HTML report is retained.

Compare candidate against its matched control **and** the existing 24.67% held-out joint mAP. Inspect Rank-1, false reconnects, HOTA/sMOTSA and skipped AMP updates. Better mAP alone does not establish safer recovery or preserve segmentation; do not promote the global recovery mode from this small diagnostic. If neither arm improves, close this test and proceed to the planned RF-DETR Seg comparison rather than adding another sweep.

AdaSP and classification are training-only. The saved tensor keys and deployed shared ViT-L / Co-DINO / RefineMask / 256D ReID architecture remain unchanged. The common trainer's `loss_triplet` dictionary key holds AdaSP for candidate VRAI updates; the logged VRAI loss is classification plus the selected metric loss.
