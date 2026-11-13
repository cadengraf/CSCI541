# Transfer specialist ReID knowledge into the joint model

This experiment continues the trained **Co-DINO/RefineMask joint model** using
the selected VRAI-fine-tuned TransReID specialist as identity supervision. It
keeps the shared detector, instance masks, mask-guided pooling and 256D embedding.
This is knowledge transfer into the existing large joint architecture; it does
not create a compact deployable model or an `.rpk` file yet.

## Run

From `capstone/reid`, with the PC `.venv` active:

```bash
python train.py joint-distill
```

For an SSH run:

```bash
tmux new -s joint-distill
python train.py joint-distill
```

Detach with **Ctrl+B**, then **D**. Reattach with
`tmux attach -t joint-distill`. The same training command resumes the last saved
optimizer step. It saves every 250 KITTI pair steps and at each epoch boundary;
an interrupted run may lose work since that save. All recovery files use atomic
replacement. Do not run two writers against the same checkpoint/cache paths.

Settings are in [`configs/distill_joint.yaml`](../configs/distill_joint.yaml).
The command automatically prepares frozen teacher targets in the modern PC
environment, releases the specialist, and launches joint training in the isolated
legacy teacher environment. Neither environment needs to load both networks at
once. Target creation resumes completed chunks. Short GPU checks do not write
training checkpoints:

```bash
python train.py joint-distill-check
```

Training automatically evaluates after all eight epochs. To rerun evaluation:

```bash
python benchmark.py joint-distill
```

## What is learned

| Step | Input | Losses |
| --- | --- | --- |
| KITTI | Two adjacent labeled car frames | Existing box/classification, instance mask, mask-quality, identity classification and triplet losses |
| VRAI | Four physical vehicle IDs, two cameras per ID | Identity classification, triplet, specialist alignment and relational matching |

A VRAI update follows every KITTI pair update. The first epoch freezes the ViT
backbone; the following seven unfreeze its last four blocks. Detection, mask and
ReID heads remain trainable, as in the original joint training path. Smaller
learning rates, gradient clipping, checkpointed activations and mixed precision
in the backbone limit the size of updates and GPU memory use.

The specialist produces normalized **3,840D** embeddings. The joint model produces
normalized **256D** embeddings. We do not compare those incompatible vectors
directly:

1. A **training-only linear projection** maps 256D to 3,840D. Cosine alignment
   encourages its output to agree with the frozen specialist target.
2. A **pairwise cosine loss** matches the similarity relationships among the
   joint embeddings to those among the specialist embeddings. Positive and
   negative ID pairs receive equal aggregate weight; diagonal self-pairs are
   excluded. This directly supervises the geometry of the 256D space used at
   inference, rather than relying only on the projection.
3. The original identity classification and triplet objectives remain active.

The objective adds `projection_weight × alignment + relation_weight × relation`
to the original VRAI identity loss. Initial weights are **5 and 10**. Gradients
flow through the joint ReID branch, predicted soft masks, shared encoder/neck and
the unfrozen backbone blocks. The teacher is frozen. The projection is saved for
training recovery but is not part of the joint model's inference state.

Targets come from clean canonical vehicle crops. Joint crops receive the existing
color, grayscale and flip augmentations while targeting those clean embeddings;
this asks the identity representation to tolerate those changes. The specialist
uses its own 256×256 preprocessing; the joint model retains its original crop and
full-frame preprocessing. There is no ground-truth VRAI mask supervision: KITTI
continues to supply real instance masks.

## Data and checkpoint safeguards

- Start from the pinned best completed joint checkpoint, including its existing
  identity classifier. Original model weights are never used as output paths.
- Pin the selected specialist's weight SHA-256. Its targets cover **only the
  52,966 VRAI training images**, with **5,042 IDs**.
- Check that the joint and specialist training manifests match, that cached paths
  cover exactly the training set, and that train/calibration/final IDs do not
  overlap.
- Cache normalized targets as float16 tensors in `data/distillation/transreid_vrai/`.
  Chunk hashes and a completion manifest prevent using incomplete/corrupt caches.
  These are internal `.pt` tensors, not image, JSON or NPZ dumps.
- Save model, projection, optimizer, AMP scaler, sampler and RNG states. Recovery
  rejects changed training settings, teacher provenance, identity mappings or
  VRAI manifest hashes.
- Preserve the starting joint model as an epoch-zero best candidate. Select new
  checkpoints by **all calibration identities' mAP**, never final-query results.
  The original joint run used a 128-ID calibration subset, so its calibration
  score should not be compared directly with this run's full-calibration score.

Each epoch samples 4,050 KITTI pairs and 32,400 VRAI crops. Sampling is identity
balanced and does not guarantee visiting every training image in each epoch.

## Outputs

| Path | Contents |
| --- | --- |
| `checkpoints/distilled_joint/joint.last.pt` | Resumable training state |
| `checkpoints/distilled_joint/joint.best.pt` | Best full-calibration joint checkpoint |
| `metrics/distilled_joint/training.csv` | KITTI loss, VRAI loss, alignment, relation and calibration scores |
| `metrics/distilled_joint/summary.html` | Final local VRAI retrieval and KITTI MOTS validation |
| `metrics/distilled_joint/vrai.csv` | Mask/global pooling ablation on the same checkpoint |
| `metrics/distilled_joint/kitti_mots/summary.html` | Official TrackEval scores for the baseline tracker |

The existing baseline tracker is used for this automatic evaluation. Do not mix
these scores with McByte++/EdgeTAM results without rerunning the same tracker on
both models. The specialist and alignment projection do not run during joint
inference. Inference architecture and embedding dimension stay the same, but
accuracy and timings still need to be measured after training.

## Matched control

To distinguish distillation from simply training the joint model longer:

```bash
python train.py joint-distill-control
```

This uses the same start checkpoint, data, sampling seed, augmentation and eight
epoch budget, with both distillation weights set to zero. The projection is still
initialized so model initialization consumes the same RNG draws. Ordinary
identity and KITTI losses remain active. It saves to separate
`checkpoints/distillation_control/` and `metrics/distillation_control/` paths.
Run the two experiments sequentially on the same GPU. Control evaluation can be
repeated with `python benchmark.py joint-distill-control`.

The primary comparisons are joint ReID Rank-1/mAP, mask versus global pooling,
and mask/tracking preservation relative to the starting model and this control.
Retaining KITTI supervision does **not guarantee** detection or mask accuracy
will be preserved. Inspect HOTA, sMOTSA, mask-based recall and MOTSP as well as
ReID metrics. If the new checkpoint loses segmentation quality, checkpoint
selection or optimization will need adjustment using separate calibration data.

The local final retrieval set has already been reported in previous project
experiments; it is not a new blind test. Published leaderboard claims still
require an official protocol or a genuinely untouched evaluation set.

## Code map

All new implementation is in `utils/distillation/`:

- `settings.py`: experiment config and separate control outputs.
- `targets.py`: frozen specialist target generation, audit, hashes and resume.
- `data.py`: camera-balanced student crops aligned with teacher targets.
- `losses.py`: alignment, relational loss and gradient/geometry checks.
- `training.py`: joint continuation, recovery and calibration selection.
- `command.py`: modern-environment caching and isolated joint-environment launch.

Existing teacher loaders, model, KITTI losses and evaluation are reused. The short
check uses crowded real KITTI frames, both backbone phases, and specialist-guided
VRAI updates, including gradient flow through the mask head. No compact student
or temporal memory distillation is included in this experiment.
