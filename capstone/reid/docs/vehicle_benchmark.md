# Quantitative vehicle benchmark

Run from `capstone/reid`:

```bash
../.venv/bin/python benchmark.py vehicle
```

The current separated-tracklet comparison uses `configs/vehicle_benchmark_tracklets.yaml` and writes to `metrics/vehicle_benchmark_tracklets_stable/`. The original v1 configuration/results remain available in `configs/vehicle_benchmark.yaml` and `metrics/vehicle_benchmark/`. See [current architecture](archive/vehicle_tracklets.md). Device selection is automatic. The command resumes completed/partial extraction and tracking, scores all nine conventional KITTI MOTS validation sequences, runs the four-model cross-camera comparison, and builds `metrics/vehicle_benchmark/scorecard.md`. No frame count, source path, output, or device flags are required.

For a long job:

```bash
tmux new -s vehicle_metrics '../.venv/bin/python benchmark.py vehicle > /tmp/vehicle-metrics.log 2>&1'
```

Rerun the same command after interruption. Dependencies are in `requirements.txt`; the current environment has been prepared. Official ByteTrack and OC-SORT sources are pinned in YAML and fetched automatically if absent. Runtime adapters observe detections and carry their masks without changing the upstream association algorithms. Source changes are rejected.

## What is actually compared

The fixed-detector experiment uses the existing complete Co-DINO prediction cache for all nine validation sequences: 2,981 frames and 8,029 evaluated car annotations. This controls detection and segmentation while testing association. It differs from the default Mask R-CNN demo backend.

- Official ByteTrack association with the common vehicle detections/masks.
- Official OC-SORT association with the same inputs.
- Existing spatial/ReID tracker with the joint model's features.
- Existing spatial/ReID tracker with independent TransReID features.
- Persistent memory with joint features.
- Persistent memory with independent TransReID features.
- Persistent memory with quality/memory-write gates disabled.

The feature/memory ablations use fixed initial thresholds; the joint features are not independently recalibrated. They isolate implementation behavior at these settings, rather than establishing the best achievable operating point for every feature space. ByteTrack/OC-SORT retain their track-confirmation and output suppression behavior. All methods receive the same raw candidate cache, including the low-confidence candidates needed by two-stage association. The appearance-based methods use the configured 0.3 confidence threshold.

The current source detection, rather than its smoothed Kalman box, supplies ByteTrack's output mask. No predicted mask is fabricated, no output is retroactively relabeled, and mask-adapter drops must be zero. Unknown identity decisions from the new memory tracker become unique one-frame IDs in MOT scoring: detection recall is preserved, and identity fragmentation is penalized. They remain unknown for recovery evaluation. Omitting rejected detections would instead conflate identity rejection with detector recall.

The previous McByte++/EdgeTAM results remain available in `metrics/tracking/summary.csv`. That implementation finalizes exported identities after the sequence. They are context, not a causal baseline in the new headline table. ByteTrack and OC-SORT are established baselines; this suite does not establish competitiveness against every current tracker or reproduce the original papers' detector setups.

## Metrics and outputs

`summary.csv` and the per-method detailed files come from the repository's pinned TrackEval implementation: mask HOTA, DetA, AssA, IDF1, sMOTSA/MOTSA, and ID switches. `paired_intervals.csv` resamples whole sequences and reports the paired mean per-sequence HOTA delta, with a 95% bootstrap interval. This mean-sequence statistic differs from pooled HOTA. The sample is nine sequences, so interpret intervals accordingly.

`recovery.csv` reports natural GT visibility gaps separately from normal association. A return event starts when a previously visible physical object reappears after missing annotation frames. It succeeds only if the pre-gap predicted identity is emitted within ten visible observations. Report all-return recovery and conditional recovery among events with a prior matched reference; methods that never enroll difficult objects cannot improve by shrinking their denominator. Events are grouped by missing-frame counts 1–5, 6–15, 16–60, and 61+. The current nine sequences contain only 26 natural return events; this is inadequate to establish unrestricted long-term recovery. No synthetic disappearance is mixed into these counts.

`cross_camera.yaml`/`.csv` re-execute the existing compact model, MSINet, pretrained TransReID, and fine-tuned TransReID on the same fully audited local VRAI split. In addition to closed-set Rank-1/mAP, they use one reference per known identity and withhold every fifth sorted identity as unknown. Thresholds and margins maximize correct known acceptance under empirical 1% and 5% calibration budgets for both unknown false acceptance and wrong known acceptance. They are then frozen for final identities. Report achieved final error, not merely the requested calibration budget. Confidence intervals resample identities, preserving correlated queries from the same identity; identity-level any-query false-accept risk also has a one-sided upper confidence bound.

`scorecard.md` puts these results and limitations in one place. `run.yaml` records the detector cache provenance, source revisions, model/configuration conditions, and evaluation revision. Small MOTS prediction text files and working feature/state checkpoints are necessary for repeatable evaluation; no per-frame image, JSON, or NPZ dumps are generated. Working checkpoints live under `data/vehicle_benchmark/` and are ignored by Git. Use new prediction/feature paths if changing model settings; the old reported experiment remains intact.

Association replay and specialist extraction timings are labeled separately. They cannot establish live end-to-end FPS. A live speed claim requires detector, segmentation, specialist, quality checks, association, and rendering on the same hardware, including periodic I/O. No GPU historical detector latency is added to CPU replay latency.

## What the first measurement says

The first full comparison identifies a substantial failure in the new persistent association: it fragments many identities that the existing tracker maintains. This is visible in AssA, IDF1, ID switches, and the rejected-observation counts despite nearly unchanged DetA. It is not evidence for replacing the specialist or adding a larger segmentation model. Short videos concealed the failure.

Use the measured existing tracker + specialist as the current identity baseline. The next association implementation should preserve local tracklet continuity while treating lost-identity reconnection as a separate open-set decision. Evaluate changes using training/development sequences, then freeze settings before rerunning a held-out comparison. Do not repeatedly tune on the nine reported sequences and call them an untouched test set.

The local VRAI crop split is an identity-disjoint split of official training images. It cannot be ranked against published official VRAI test results; this repository does not have identity labels for that hidden protocol. Likewise, KITTI validation scores cannot be compared numerically against hidden-test leaderboard rows. External benchmark claims require the matching dataset and protocol.

Protocol references: [KITTI MOTS](https://www.cvlibs.net/datasets/kitti/eval_mots.php), [TrackEval](https://github.com/JonathonLuiten/TrackEval), [ByteTrack](https://github.com/FoundationVision/ByteTrack), [OC-SORT](https://github.com/noahcao/OC_SORT).
