# Controlled vehicle appearance training

From `capstone/reid`, run or resume the whole queue with:

```bash
../.venv/bin/python train.py appearance
```

The active detached job is `appearance_training`; attach with `tmux attach -t appearance_training`. Its log is `/tmp/appearance-training.log`. To launch a new detached queue when one is not already running:

```bash
tmux new -s appearance_training '../.venv/bin/python train.py appearance > /tmp/appearance-training.log 2>&1'
```

Settings: `configs/appearance_training.yaml`. A file lock prevents duplicate queues. Checkpoints resume automatically; no `--resume` flag is needed. Configuration, code, model, and manifest signatures reject incompatible resumes. Stop the worker before editing signed experiment modules or settings; use fresh output/checkpoint/state paths for intentional changes. The original specialist, its selected weights, and video defaults are never overwritten.

## First experiment queue

All specialist pilots start from the existing VRAI-adapted TransReID checkpoint and run the same 3 epochs × 400 steps, with 8 identities × 4 images per batch and two cameras per identity. These are initial continuation pilots, not convergence claims.

1. Frozen specialist mining: extract up to two crops per training camera and identity. Build cross-camera mean prototypes; select 20 high-similarity *different labeled training identities* for each identity. Similarity proposes training examples, never identity labels. The graph stays fixed for controlled comparisons. Only training identities are used.
2. Continuation control: existing augmentation, random identity sampling, existing CE + triplet loss.
3. Degradation: add moderate crop extent/translation changes, downsampling, blur, and JPEG compression to half the images. Keep existing augmentation and a mixture of clean examples. These are detector-like perturbations of VRAI crops, not actual detector outputs.
4. Hard sampling: put one anchor and three sampled neighbors into each eight-identity batch; fill the remainder with random identities. Preserve four observations and two cameras per identity. No synthetic identities or pseudo-labels are introduced.
5. AdaSP: use the original sampling/augmentation and replace the metric term with AdaSP. Its loss values and gradients were checked against the vendored CUDA implementation. The adapter supports arbitrary label order and device-safe tensors; balanced identity groups are required.
6. Conditional combination: combine degradation and hard sampling only if *each* independently improves calibration-selection correct known acceptance by at least one percentage point over the continuation control, passes the empirical error budgets, and loses no more than 0.5 points of full calibration mAP. Otherwise explicitly skip it.
7. CLIP-ReID: use the pinned implementation already in the repository and checksum-verified OpenAI CLIP ViT-B/16 initialization. Learn identity prompts for 600 steps from frozen training-image features, freeze their text features, then fine-tune the image encoder for 1,200 steps. Camera/view side embeddings are disabled. The prompt stage adds a separate budget; CLIP starts from a different initialization than the already-adapted TransReID. Its pilot cannot establish a fair architecture ranking or convergence.

There is no detector, segmentation network, temporal propagation, or tracking association in these training jobs. At inference the TransReID specialist supplies image embeddings; classifiers and metric losses are training-only. CLIP uses its image encoder and output features for retrieval; prompts/text features guide training and are unnecessary for retrieval inference. These training runs do not measure online tracking FPS.

## Evaluation and checkpoint selection

Train on the existing 5,042 VRAI training identities (52,966 crops). Use only the existing 630 calibration identities for all pilot decisions:

- Even positions in sorted calibration IDs: 315 threshold-fitting identities.
- Odd positions: 315 checkpoint-selection identities, separate from threshold fitting.
- Fit score/margin thresholds to maximize correct known acceptance under 1% empirical unknown-FAR and wrong-known-query-rate budgets. In each subset every fifth sorted identity is withheld as unknown; use one first-manifest gallery reference per known identity.
- Score these fixed thresholds on the checkpoint-selection subset. Prefer empirically eligible checkpoints, then correct known acceptance, lower errors, and calibration mAP as a tie-breaker. The small subsets limit precision of these estimates; eligibility is not a statistical risk guarantee.
- Report Rank-1/mAP on the full calibration query/gallery for continuity with the prior specialist report. Starting scores reproduce 94.06% Rank-1 / 93.40% mAP.

The existing final VRAI query/gallery images are not evaluated by this queue. Their manifests are checked for train/calibration separation, not used to select a checkpoint. These runs make no new KITTI tracking or cross-camera deployment claim. The next integration is to assess a selected encoder with the same fixed detector and association, rather than changing multiple components together.

## Outputs

- `metrics/appearance_training/queue.yaml`: running/current stage or failure, full experiment signature.
- `metrics/appearance_training/summary.md`: completed/partially completed pilot results.
- `metrics/appearance_training/<variant>/calibration.csv` and `report.yaml`: checkpoint-selection results and training progress.
- `metrics/appearance_training/hard_negative_review.mp4`: train-only visual inspection of anchor vehicles and confusing labeled alternatives. Sampled triplets were visually inspected; this does not certify every dataset label.
- `checkpoints/appearance_training/<variant>/{last,best}.pt`: latest full optimizer/scaler/RNG state and selected model weights. `best` can remain the starting checkpoint when continuation does not help.
- `data/appearance_training/`: resumable mining/prompt working state and isolated tokenizer dependencies. Completed TransReID mining discards per-crop embeddings, retaining the neighbor graph. CLIP's frozen training features are needed for prompt training/resume. No per-image feature, JSON, or NPZ exports are produced.

Checks executed before launch: six CPU tests; two real CUDA updates each for continuation, degradation, and AdaSP; AdaSP value/gradient agreement with the vendored reference; two real CUDA updates in each CLIP training stage and finite 1,280-dimensional inference embeddings. Original video-association tests remain separate.

## Completed pilot results

All training stages are now complete, including CLIP-ReID after correcting its namespace-package import guard. Combined training was skipped because its independent-improvement criteria were not met. The prompt checkpoint and frozen training features were preserved, backed up, and tensor-verified during the guard-only provenance migration.

See [executed analysis](../../metrics/appearance_training/analysis.md), [selected checkpoints](../../metrics/appearance_training/summary.md), and [48-pixel query diagnostic](../../metrics/appearance_training/robustness/summary.md).

The useful finding is resolution robustness: final-trained degradation reaches 93.53% Rank-1 versus original 87.86% on the same 315-identity calibration-selection gallery with 48-pixel queries. Correct known acceptance rises from 69/498 to 133/498; unknown false accepts rise from 0/120 to 1/120. Against equal-budget continuation, degradation improves known acceptance from 89/498 to 133/498 at the same 1/120 unknown error count. Clean-image acceptance and mAP decline. The diagnostic retains clean fitted thresholds and uses no final evaluation images. It is not KITTI video evidence.

For this candidate, use the model state in `checkpoints/appearance_training/degradation/last.pt` (epoch 3); its `best.pt` intentionally remains the starting epoch-0 specialist under the original clean selection objective. No deployment selection was overwritten.
