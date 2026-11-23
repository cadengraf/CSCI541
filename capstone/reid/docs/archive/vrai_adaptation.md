# VRAI-only adaptation of the combined model

This experiment asks whether the existing joint network can learn stronger
cross-camera identity features when all training updates focus on VRAI.

## Run

From `capstone/reid`, with the PC virtual environment active:

```bash
python train.py vrai-adapt
```

For SSH:

```bash
tmux new -s vrai-adapt
python train.py vrai-adapt
```

Detach with Ctrl+B, then D; reattach with `tmux attach -t vrai-adapt`.
The same Python command resumes checkpoints. Saves occur every 100 VRAI updates
and at epoch boundaries, so interruption can lose up to 100 updates. Do not run
two writers against these paths. A completed training run is skipped on rerun,
and evaluation runs afterward.

```bash
python train.py vrai-adapt-check   # one real GPU update; no checkpoint writes
python benchmark.py vrai-adapt     # reevaluate completed weights
```

Configure `configs/vrai_adaptation.yaml`. Long training is not started by checks.

## Training

- Start from the pinned **square-256 crop-training best checkpoint**, selected
  epoch 4: 22.56% Rank-1 / 26.15% mAP on full VRAI calibration.
- Keep the combined Co-DINO/RefineMask/256D ReID architecture and classifier.
  No spatial adapter is added.
- Sample **four VRAI vehicle identities × two cameras**, eight crops per update,
  from 52,966 images / 5,042 training identities. Existing color/grayscale/flip
  augmentations remain active. Identity crops use square 256 preprocessing.
- Optimize the existing identity classification and batch-hard triplet losses.
  No new teacher distillation or direct box/mask/quality losses.
- **No KITTI image forwards or KITTI identity training updates.** The KITTI data
  loader is still constructed to preserve the original sequence-local identity
  mapping/classifier layout and to enable evaluation. VRAI labels retain their
  original classifier offset. Unused classifier classes remain negative logits;
  the output classifier is not replaced.
- Train eight epochs × 2,000 VRAI updates = **16,000 scheduled optimizer updates**,
  128,000 crop presentations, not 128,000 unique images. Repeated sampling means
  this is not guaranteed exhaustive coverage of the dataset.
- Train the last four ViT blocks and the ROI/encoder/ReID path, with head LR 1e-5
  and backbone LR 1e-6, AdamW, cosine decay, clipping and AMP. This does not train
  every backbone block. Parts with no gradients naturally do not update.

The predicted soft mask still guides pooling, so identity loss can change the
mask branch. Shared feature changes can degrade detection/segmentation even
without direct gradients to detector prediction layers. The subsequent KITTI
check is essential before using these weights for the full system.

## Selection and evaluation

Full VRAI calibration mAP selects the best checkpoint, including epoch zero as
a fallback. Initial calibration is measured before any adaptation. After all
epochs, the selected checkpoint is evaluated with mask and global pooling and
with the same baseline tracker on conventional KITTI validation. Full-frame
preprocessing remains unchanged. VRAI final-test embeddings are not extracted.

Calibration is development data used for earlier decisions and checkpoint
selection. These scores are not a new blind test. This run has a different
identity exposure and total update budget from the earlier mixed KITTI/VRAI runs;
a gain cannot by itself be attributed solely to removing KITTI objectives. It
measures the architecture's adaptation capability under this training schedule.
One seed provides preliminary evidence.

## Outputs

- `checkpoints/vrai_adaptation/vrai_only.{last,best}.pt`
- `metrics/vrai_adaptation/vrai_only/training.csv`
- `metrics/vrai_adaptation/vrai_only/summary.html` — retrieval, tracking and timing
- `metrics/vrai_adaptation/adaptation.html` — origin/selected/last comparison
- `metrics/vrai_adaptation/summary.csv` — calibration summary

Weights, optimizer and AMP state are saved atomically. Per-update sampling and
torch augmentation seeds are derived from epoch/step, so resuming does not
change subsequent sampled batches; GPU execution is not guaranteed bitwise
identical. Recovery rejects changed settings, identity mappings or manifest
hashes. Existing model checkpoints are never output destinations.
