# Temporal evidence recovery experiment

From `capstone/reid`:

```bash
../.venv/bin/python benchmark.py vehicle-evidence
```

The command resumes checkpointed development and validation replays, then scores committed identities with TrackEval. It requires the existing completed specialist banks and frozen detector cache. Settings are in `configs/tracklet_evidence.yaml`. `device: auto` remains the inference default; this experiment only replays existing embeddings and requires no GPU model inference.

For a detached job:

```bash
tmux new -s vehicle_evidence '../.venv/bin/python benchmark.py vehicle-evidence > /tmp/vehicle-tracklet-evidence.log 2>&1'
```

The four variants use the same local tracker, detections, masks, and trained specialist:

- Frame control: original three-consecutive-claim global recovery.
- Query mean: pooled evidence from three reliable observations, at least two frames apart and spanning four frames.
- Dual pool: query pooling plus frozen enrollment consensus. The original reference embedding is retained. Both reference and source consensus must meet the reference floor.
- Dual quarantine: dual pooling, with the trusted recent bank frozen after any global recovery. A recovered tracklet cannot rewrite its trusted appearance evidence.

Pooling weights the normalized specialist embeddings by the square root of original box area. It rejects a pool if any sample has cosine below 0.85 to the mean. An unreliable observation or visibility gap resets incoming evidence. A source pool can only be created during its original enrollment tracklet and view epoch. Frame/camera resets clear incoming temporal state and preserve reference evidence. Every operation uses current/past frames only. Pooling and all matching/memory logic run during inference; no extra training branch or loss is introduced by this experiment.

Pooled variants can act after five consecutive reliable observations; the control can act after three. This is an accuracy/latency tradeoff. Pooling reduces noise but cannot recover fine detail missing from tiny crops. Reduced claims must be read alongside correct recoveries and natural-return coverage.

All 36 threshold/variant combinations are assessed on training-partition development sequences 0004/0005/0011/0020. Each variant needs at least 20 scored global claims, no unscored claims, and at most 1% empirical wrong claims to qualify. Selection is committed before validation. If none qualifies, validation assesses the pre-registered 0.82 similarity / 0.08 margin setting as a diagnostic. Validation never selects a threshold. No variant automatically replaces the default reference engine.

`metrics/vehicle_tracklet_evidence/summary.md` reports global claim correctness, natural visibility returns, HOTA, IDF1, ID switches, and paired sequence uncertainty. Local reacquisition errors remain separate in `report.yaml`. Replays are not end-to-end inference speed measurements and are not cross-camera tests. Cached extraction time is not counted as online FPS. The four development control identity streams must exactly reproduce the earlier experiment before validation proceeds.

The false-claim inspection video is `metrics/vehicle_recovery_review/development_global_review.mp4`. It displays the original enrollment, query, and recent evidence for all six original development false claims. Its annotations are evaluation evidence only and never enter association.

Working checkpoints and evaluator MOTS inputs remain under `data/vehicle_tracklet_evidence/`; there are no per-frame image or embedding exports. Inputs/code hashes reject incompatible resumes. For code or protocol changes, use a fresh state root and output directory in a YAML configuration. Prior signed results remain unchanged.

## Executed result

None of the 36 development points qualified. At the original thresholds, development correct/wrong global claims were 0/6 (control), 0/9 (query mean), 0/15 (dual pool), and 0/16 (dual quarantine). Frozen validation query pooling reached 79.616 HOTA / 87.571 IDF1 versus reference 79.456 / 87.346; its paired sequence HOTA interval includes zero. It accepted 6 correct / 1 wrong / 2 unscored global claims. This does not establish a reliable operating point. The default reference is unchanged.

The query-only configuration is `configs/vehicle_reid_query_pool.yaml`. A live 100-frame run produced `data/videos/vehicle_query_pool/annotated.mp4` at 7.20 FPS on RTX 4090, including all online components and checkpoint/video I/O; initialization was 7.23 seconds separately. To replay its completed video job, use `../.venv/bin/python demo.py vehicle --config configs/vehicle_reid_query_pool.yaml --resume`. It remains experimental.

A separate 25-frame restart smoke test imported the saved target, discarded spatial state, and emitted one recovered-target action. Its output is `data/videos/vehicle_query_pool_profile/annotated.mp4`. This same-video integration check does not measure cross-camera accuracy.
