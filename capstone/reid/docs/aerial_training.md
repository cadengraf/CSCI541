# Balanced aerial vehicle training

From `capstone/reid`, run or resume:

```bash
../.venv/bin/python train.py aerial
```

The authorized detached queue runs in tmux session `aerial_training`. Its `logs` window follows `/tmp/aerial-training.log`:

```bash
tmux attach -t aerial_training
```

Configuration is `configs/aerial_training.yaml`. Device is always automatic. The queue advances through all variants without another conversation turn. Checkpoints include optimizer, AMP scaler, identity mapping, both sampling RNGs, and Torch/NumPy/Python/CUDA RNG state. Rerunning the same command resumes; a lock rejects duplicate queues. Signatures protect against changed code/configuration/manifests. Intentional experiment changes need new output/state/checkpoint paths.

## Data before training

The existing train manifests contain 52,966 VRAI crops / 5,042 IDs and 80,532 VRU crops / 7,085 IDs. Reserve 708 VRU training identities for development, split into 354 rejection-fit and 354 checkpoint-selection IDs. Preserve the previous VRAI calibration identities and all existing test manifests. Dataset prefixes distinguish IDs such as `vrai:123` and `vru:123`.

The pixel/file audit examined 231,922 unique image paths, reading existing test images only to check duplicate leakage. It caches file SHA256, decoded RGB pixel SHA256, a small perceptual fingerprint, image geometry, and gray thumbnail values in SQLite. No test images are used for training, checkpoint choice, threshold fitting, or performance evaluation.

Training identities with exact duplicate pixels under conflicting labels or duplicate held-out evidence are quarantined. Within-identity exact duplicate crops are removed. Near duplicate train/held pairs use an exact perceptual fingerprint plus very close aspect ratio and gray thumbnail agreement; suspected training identities are quarantined instead of automatically merging labels. This cannot establish physical identity separation for every independently collected scene.

**Final training pool: 125,328 crops / 11,407 identities:**

| Source | Training crops | Training identities |
|---|---:|---:|
| VRAI | 52,821 | 5,034 |
| VRU | 72,507 | 6,373 |

The audit quarantined 12 training identities and removed 97 redundant within-identity crops. Three VRAI development identities have exact-pixel evidence in the old warm-start model's training data. Removing new training images alone would not undo that prior exposure: these held-out identities are excluded from the new development manifests as well. A fourth development identity is excluded due to conflicting held-out duplicate evidence. VRAI development therefore uses 313 fit and 313 selection IDs. No exact-pixel matches were found between the old VRAI training images and the existing final-test images. This is a pixel audit, not proof against all physical-identity or near-duplicate leakage.

Original manifests, models, and historical metrics remain unchanged. Initial prepared manifests before the extra warm-start-exposure filtering were archived under `data/aerial_training/before_prior_exposure_filter/`.

## Training program

All four variants warm-start the original specialist's encoder. The classification heads expand from 5,042 labels to 11,407 labels. Existing clean VRAI rows are remapped by identity; VRU rows receive independent initialization. Numerical label collisions across datasets never copy weights or imply shared physical identities.

Each batch has 4 VRAI identities and 4 VRU identities, with 4 observations each (32 images). VRAI observations use its two actual camera labels. VRU has no camera labels in these manifests: its sampler draws distinct photos when available and never invents camera IDs or guarantees cross-camera separation. A source having many more images cannot dominate the batch.

Each matched run receives 20 epochs × 1,500 updates (30,000 updates), with one classifier-warmup epoch followed by full-backbone fine-tuning. AdamW, encoder learning rate 1e-5, classifier rate 1e-4, cosine decay, label smoothing, gradient clipping, mixed precision, and checkpointing every 250 updates. These are substantial continuation runs; convergence is assessed from development curves rather than assumed from the epoch count.

1. `mixed_baseline`: existing crop/flip/color/erasing augmentation; classification plus batch-hard triplet loss.
2. `mixed_degradation`: moderate crop/framing variation, downsampling, blur, and JPEG perturbation on 35% of samples, mixed with the original augmentation. The short-side range is 32–160 pixels to preserve more clean-detail evidence than the earlier aggressive pilot.
3. `mixed_hard_sampling`: same augmentation/loss as baseline, using a fixed graph mined from the completed mixed baseline's selected checkpoint and training images only. Within each source, half the batches include one confusing identity pair; other identities remain random. Positives retain their known identity labels. No test or development crops enter mining.
4. `mixed_adasp`: baseline data/sampling, replacing the metric term with AdaSP. The reused AdaSP implementation has already passed loss/gradient checks against its vendored reference.

Augmentation and sampling use independent RNG streams, so degradation does not silently change the control's identity/image choices. All variants restart from the same original specialist and classifier initialization; they do not sequentially fine-tune each other's weights.

Current inputs are labeled vehicle crops. Framing augmentation approximates detector errors; actual detector-produced aerial video crops have not yet been added to this training pool. That needs a separately audited video-crop preparation stage. This program introduces no detector, mask, motion, or geometry training.

## Development selection and baseline

Select checkpoints by the equal-weight mean of VRAI and VRU **development** mAP, with macro known acceptance as a tie-breaker. Each source retains its own gallery protocol, so this macro value is a training objective, not a new unified benchmark score.

Fit score/margin rejection thresholds on the fit identities; report correct known acceptance, wrong-known acceptance, and unknown FAR on separate selection identities. Rejection diagnostics do not automatically enable persistent matching or establish a population error guarantee. No full VRU or VRAI test evaluation is performed by this queue.

The existing VRU-trained MSINet checkpoint is checksum-verified and recorded as a strong archived operational reference (89.93% Rank-1 / 93.77% mAP on the prior full VRU split). It may already have seen the VRU train identities reserved for the new development split, so it never participates in development checkpoint/threshold selection. Its training conditions differ from these mixed-domain runs.

## Outputs

- `metrics/aerial_training/data_audit.yaml`: data counts, exclusions, duplicate evidence, and audit provenance.
- `reference.yaml`: frozen MSINet reference and training-condition caveat.
- `queue.yaml`: active/completed/failed stages and exact training signature.
- `summary.md`: development results as epochs finish; a lightweight background reporter refreshes it during the launched queue.
- `<variant>/development.csv` and `report.yaml`: per-epoch source-specific retrieval and rejection diagnostics.
- `checkpoints/aerial_training/<variant>/{last,best}.pt`: resumable state and development-selected weights.
- `data/aerial_training/`: separate prepared manifests, cached image audit, and train-only mining state.

Five new CPU tests verify balanced sampling, namespacing, camera honesty, identical control/degradation image selection, hard-neighbor boundaries, pixel hashing, and classifier remapping. GPU checks execute both warmup and full-backbone updates on real mixed-domain crops for baseline, degradation, and AdaSP. No video default or original checkpoint is replaced by this program.
