# Full CityFlowV2 validation benchmark

Run or resume from `capstone/reid`:

```bash
../.venv/bin/python benchmark.py cityflow
```

After you start the queue, watch it with:

```bash
tmux attach -t cityflow
```

**Run ownership:** you launch and resume long runs. The assistant must not launch, resume or schedule them without a new explicit instruction. The previously assistant-started queue is stopped, with 8,109 original-model S02 frames and 16,200 S05 frames committed. The command above resumes those checkpoints when you choose to run it.

Configuration is `configs/cityflow.yaml`; the device is automatic. Once you start the queue, all three frozen encoders advance sequentially: original VRAI TransReID, mixed VRAI/VRU AdaSP TransReID, and the author clean-VRU MSINet checkpoint. The queue defaults to the **complete official validation split**, with no frame stride, sampled identities, selected target, synthetic blackout, or camera limit: S02 and S05, 23 cameras, 81,954 video frames, 834,526 raw supplied detection rows. The installed dataset lives read-only at `/home/megrad/Downloads/AICity22_Track1_MTMC_Tracking`.

## What the earlier run showed

The small VisDrone run verified that the temporal resolver could restore original identities after some expired tracks: 2/3 artificial observation blackouts versus 0/3 for the local reference, and one extra natural recovery. It isolated appearance/association using GT boxes. It did not establish detector performance, cross-camera robustness, official benchmark accuracy, or statistically reliable false-reconnection rates. Its milestone gate remains open. That diagnostic is preserved rather than presented as a full benchmark.

## What this full run measures

CityFlowV2 supplies vehicle identities across ground cameras, videos, timestamps, ROI masks, predicted detections and its MTMC evaluator. The local complete **validation** score uses the supplied evaluator's IDF1, ID precision and ID recall. This is not an official hidden-test submission or a leaderboard result. It is ground-camera transfer, distinct from aerial accuracy.

The inference path is supplied YOLOv3 detections → confidence/size/ROI/NMS filtering → native RGB crops → frozen ReID encoder → per-camera local tracking → causal appearance association across cameras → official prediction TXT output. GT identity boxes enter scoring and sparse claim diagnostics only, never prediction. No Mask R-CNN model or segmentation stage runs. Supplied ROI images are standard dataset inputs.

Each model uses the same policy and detections, its own native normalization (256-square RGB), and its previously frozen common-VRAI score/margin calibration. No target labels fit thresholds or select checkpoints. The temporal association policy differs from the prior single-camera crop protocol: source thresholds transferring to pooled multi-camera evidence are a measured baseline, not a guaranteed target error bound. Training histories and budgets differ; this is not an architecture-only causal experiment. Joint models are excluded because historical joint training used AICity.

## Causal multi-camera baseline

Each camera retains the existing motion/IoU/appearance local tracker and its own image-coordinate state. Camera events advance in timestamp order. Same-time camera events are ordered deterministically. Spatial distance is never compared across camera coordinate systems. There is no homography, camera-link graph, offline clustering, future-frame pooling or retrospective output relabeling.

Three reliable observations, spaced two frames apart and spanning four frames, form a coherent query. Uncertain crops reset the evidence window and cannot update local appearance. A global appearance bank keeps an immutable enrollment reference plus gated recent evidence. A source-fitted score and reciprocal margin authorize matching; otherwise the observation retains a camera-local identity or no confident global match. New profiles require coherent evidence and low similarity to available existing profiles. One global identity can be observed simultaneously in different cameras, but cannot own two observations in the same camera/frame. Recovery itself does not immediately write appearance memory.

This is a simple online appearance/motion baseline, not a reproduction of challenge-winning systems. The benchmark is useful even if its score is low: full-split failures are measured under the official metric rather than hidden behind a favorable clip or GT-box discovery.

## Official evaluation and compatibility

The dataset's `eval/eval.py` runs its original metric formulas and ROI / single-camera-ID / duplicate filtering. It removes identities predicted in only one camera, so purely local tracks are not rewarded as multi-camera identity matches. The source file stays unchanged in Downloads. The adapter removes its **unused** `pytrec_eval` import after verifying the symbol is not referenced, and temporarily restores NumPy's removed `asfarray` API. No metric formula is replaced. When all predictions are filtered away, the upstream empty-concatenation failure is explicitly represented as all GT missed.

The modern project environment additionally needs `motmetrics==1.4.0`, installed from `requirements/cityflow.txt`. Synthetic perfect-match tests recover IDF1/IDP/IDR = 1, and one-camera-only predictions exercise the official exclusion rule. Eight tests cover timestamp scheduling, overlapping views, ambiguous lookalikes, memory gating, pickle resume, exact seeking against a real source video, and prediction-transaction interruption/resume on real video/detection inputs with a deterministic test encoder. Resumed prediction bytes exactly match uninterrupted output; complete stages skip encoder loading. The separate GPU integration check uses all three actual encoders.

## Checkpoints, logs and outputs

`metrics/cityflow/selection.pt` and `protocol.yaml` freeze complete cameras, source counts, frame metadata, detection/ROI/GT/evaluator checksums, weights, source thresholds and code. `queue.yaml` records the active encoder and completed jobs. Every 300 processed camera-frames, committed prediction chunks and the complete tracking/appearance state are saved. On restart, readers seek to their next committed frames and the scheduler restores chronological order. A video-seek test verifies exact decoded pixels against sequential decoding. Orphan uncommitted chunks are overwritten. A queue lock prevents duplicate jobs.

No crop images or complete feature banks are exported. Required tracker checkpoints and prediction chunks live under `data/cityflow/`. Each model produces the official nine-column `predictions.txt`, a small diagnostics state, and `report.yaml` under `metrics/cityflow/<model>/`. `summary.md` and `report.yaml` update after each completed encoder. Sparse accepted-claim diagnostics use IoU >= .5 against labeled MTMC vehicles, distinguish correct/false/unscored claims, and remain separate from official IDF1. They do not silently treat unannotated detections as correct.

After all models finish, `review.mp4` shows bounded windows from **every** camera with all three models side by side. Scoring uses every frame, regardless of video review length. The review is not an accuracy sample.

Component timing covers input preparation, decode, embedding, association, initialization and checkpoint writes separately. Detector generation was performed by the dataset provider, so its cost is unavailable. The run does not claim full live-system FPS. Rendering and scoring occur afterward and are excluded from model timings. Final persisted checkpoint-write timing may lag one write; this limitation does not affect predictions or metrics.

`benchmark.py cityflow --check` is a separate integration-only run on 50 frames per camera under `check50` output/state paths. It must never be presented as full benchmark accuracy. Main reported evaluations use the full split. Short checks remain only a prerequisite to avoid wasting long GPU runs on integration errors.

Official dataset/evaluation context: [AI City 2022 data and evaluation](https://www.aicitychallenge.org/2022-data-and-evaluation/). Different splits, external training conditions, and offline/full-system methods cannot be compared to this local causal baseline as though all conditions matched.
