# Close the current joint-model experiment

Keep ViT-L / Co-DINO / RefineMask / 256D ReID for this stage. The matched distillation and no-distillation runs already finished; do not repeat them.

| Existing run | VRAI held-out mAP / Rank-1 % | KITTI HOTA / sMOTSA |
|---|---|---|
| Original joint | 12.70 / 11.86 | See original report |
| Continued training without distillation | 18.07 / 15.54 | 79.43 / 77.33 |
| Specialist-to-joint distillation | 20.47 / 17.91 | 77.56 / 76.79 |
| Later square-crop joint | Pending; calibration is 26.15 / 22.56 | 79.29 / 76.65 |

Distillation gained **2.39 mAP points**, but lost **1.88 HOTA points** versus its matched control. The square-crop continuation is a later experiment with a different training history, not another matched distillation arm. Sources: the CSVs in `metrics/{teacher,distillation_control,distilled_joint,crop_training}/`.

## Next user-launched run

From `capstone/reid`:

```bash
tmux new -s joint-eval
../.venv/bin/python -u benchmark.py joint 2>&1 | tee metrics/joint-evaluation.log
```

This evaluates the existing `checkpoints/crop_training/square_256.best.pt` on all held-out VRAI query/gallery records, using mask and global pooling, then reuses/evaluates all KITTI validation sequences. Checkpoint selection remains calibration-only. It does not train, resume the unfinished VRAI-only adaptation, or change the selected checkpoint. The held-out set has been used in previous experiments; it is not a new blind test.

Outputs remain under `metrics/crop_training/square_256/`: `vrai.csv`, `summary.html`, KITTI metrics and timing. `calibration.csv` is preserved. Repeating the command restarts retrieval and reuses completed KITTI sequences whose provenance matches; retrieval is not resumable within its extraction pass. Detach with Ctrl+B, D; reattach with `tmux attach -t joint-eval`.

## Exit gates, in order

1. **Close joint accuracy:** compare held-out retrieval, mask/global pooling and KITTI preservation. Record whether the existing joint approach helped; it need not match the specialist before moving on.
2. **Check identity recovery separately:** keep local continuity, disappeared-target recovery, false reconnections and abstentions separate. Compare identical videos, detector settings and association rules. Existing specialist recovery results do not validate the joint embedding.
3. **Measure cost honestly:** existing KITTI timings cover decode, model, association and RLE export, not annotated MP4 rendering. Cached sequences retain historical timings. Fresh end-to-end video timing is required for a live speed claim.
4. **Then test RF-DETR Seg:** first replace detection/masks while holding specialist and tracker fixed; then test shared spatial features plus a dedicated ReID adapter. Evaluate Cutie/McByte++ afterward with detector/encoder fixed. Do not start another unrelated training sweep first.

The user launches all long runs. `benchmark.py milestones` only refreshes the HTML from saved evidence.
