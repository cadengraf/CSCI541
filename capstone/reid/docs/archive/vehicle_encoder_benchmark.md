# Paired original versus degradation encoder benchmark

From `capstone/reid`, run or resume:

```bash
../.venv/bin/python benchmark.py vehicle-encoder
```

The running tmux session is `vehicle_encoder_benchmark`. Its selected `logs` window follows `/tmp/vehicle-encoder-benchmark.log`.

Configuration: `configs/vehicle_encoder_benchmark.yaml`. The candidate is the final epoch-3 degradation checkpoint, explicitly exported as inference-only weights into `checkpoints/vehicle_encoder_benchmark/`. It is **not** `degradation/best.pt`, which remained the starting encoder under clean calibration selection. The export retains the exact model tensors and training provenance, omitting optimizer and RNG objects so both installed inference environments can read it. Neither the original specialist nor its selection manifest is modified.

## Protocol

- All nine conventional KITTI MOTS validation sequences: 0002, 0006, 0007, 0008, 0010, 0013, 0014, 0016, 0018; 2,981 frames.
- Same frozen Co-DINO detections, scores, masks, crop geometry, RGB preprocessing, input resolution, quality gating, association thresholds, and causal output contracts.
- Only the encoder weights differ inside each pair. Compare original versus degradation with the reference local tracker; compare original versus degradation with query pooling.
- New candidate feature banks, prediction streams, and replay checkpoints have their own paths under `data/vehicle_encoder_benchmark/`. Every frame checks that both banks have identical detection indices and boxes.
- No threshold fitting/search or validation checkpoint selection occurs. Original reference and query-pooling identity streams must exactly reproduce all archived sequence streams.
- Ground-truth masks label observations only after all four algorithms have made their current-frame decisions. GT never enters embeddings, association, or memory updates.

## Metrics

TrackEval reports mask-based HOTA, AssA, IDF1, ID switches, and detection statistics. Whole-sequence paired bootstrap intervals describe mean per-sequence HOTA differences, not pooled HOTA. Nine sequences provide limited uncertainty evidence. The sequences have been inspected during prior architecture development; they are not an untouched final research test.

Recovery reports include all natural GT visibility returns, returns with a prior reference, recovered/missed/censored returns, recovery delays, wrong identity observations, and separate accepted global and local reconnection errors. A local tracking ID is not a claimed global match. Unscored global claims are never counted as correct.

Size diagnostics use a fixed 48-original-pixel short-side threshold. Observation/action bins use the predicted box, common to both encoders; visibility-return bins use the GT mask bounding box at the return frame. These are action and recovery diagnostics, **not** size-filtered HOTA/IDF1. All-return denominators retain returns without a previous matched reference.

Full live integration then runs each encoder sequentially through Co-DINO + RefineMask + independent TransReID + query pooling + rendering, encoding, checkpoint writes, and final video assembly on the same first 100 frames of sequence 0010. Initialization is reported separately from steady-state wall FPS. GPU compute processes before/after each run are recorded; if another job is present, the timing is marked non-isolated and must not be interpreted as a clean relative speed comparison. Offline feature extraction/association cost is not online FPS.

## Outputs

- `metrics/vehicle_encoder_benchmark/summary.md`: tracking, recovery, size, uncertainty, and live-cost summary.
- `report.yaml`: models, provenance, per-sequence recovery/actions/events, size diagnostics, and original-stream checks.
- `tracking/`: pinned TrackEval score outputs and paired-sequence diagnostics.
- `online_cost.yaml`: complete live-run cost and GPU-contention evidence.
- `comparison.mp4`: original left, degradation right, same live source frames.
- Per-encoder annotated videos and target profiles: `data/videos/vehicle_encoder_benchmark/{original,degradation}/`.

Checkpoint writes are atomic; committed prediction offsets allow interrupted association runs to truncate unfinished output before resuming. Completed compatible feature/replay stages are reused. Code/configuration/model/data signatures reject incompatible resume. For intentional implementation or protocol changes, use fresh output, state, and candidate pipeline paths. The live default remains the original reference while this comparison is assessed.

## Executed outcome

The degradation encoder did not improve the frozen video configuration. With query pooling, HOTA changes 79.616 → 78.969 and IDF1 87.571 → 86.461; both recover 9/26 returns, including 9/25 small returns. Accepted global claims change from 6 correct / 1 wrong / 2 unscored to 5 / 1 / 1. One natural return is gained and another lost (details in `integration.yaml`). The default remains the original encoder; no threshold calibration was performed.

All 48 tests pass, all original identity streams match archived results, and the inference export matches the final training model tensors exactly. The single-command resume check reused all nine feature sequences, all nine association sequences, and both completed live videos. See [analysis](../../metrics/vehicle_encoder_benchmark/analysis.md) and [comparison video](../../metrics/vehicle_encoder_benchmark/comparison.mp4).
