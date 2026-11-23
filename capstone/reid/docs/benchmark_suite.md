# Benchmark, compare, then change the model

With `.venv` active, from `capstone/reid`:

```bash
tmux new -s reid-bench
python -u benchmark.py suite
```

**One scorecard: [milestones.html](../milestones.html).** It contains local results, previous/best comparisons and published reference tables. It changes only when a comparable best improves. Runs without improvement print their latest/previous/best comparison in the terminal and update `metrics/scorecard/scores.yaml`; the HTML stays unchanged. Best is per benchmark, not an overall model winner.

- **ReID:** primary metric is held-out mAP; Rank-1 stays visible. Calibration never promotes a best model.
- **Recovery:** more recovered returns must not increase false reconnects. Natural/injected returns and specialist/joint results stay separate.
- **Tracking:** KITTI HOTA and CityFlow IDF1 are separate benchmarks.
- **Speed:** lower latency is compared only within matching hardware/work-unit/scope groups; encoder latency is not full video FPS.

## Next experiments

Edit `configs/benchmark_suite.yaml`: give the next experiment a new `run`, `joint_checkpoint`, and matching `joint_last_checkpoint`. Then run the same command. Use checkpoints from one completed run. Only latest, previous and best summary records are retained for each model role/test protocol; old best weights are referenced, not copied. Keep those trained checkpoint files if you need to reproduce the best.

The suite runs joint VRAI/KITTI, aerial specialist retrieval, VisDrone transfer/rejection, specialist comparisons/timing, specialist and joint recovery, and CityFlow. Disable unwanted stages in YAML before starting; dependents of incomplete/disabled stages are blocked. No training runs. No separate suite report or per-run HTML archive is retained.

Run the same command to resume. Failed/incomplete runs retain their resumable state. Completed stages are reused only if their recorded evidence still matches. Joint retrieval restarts its extraction pass; joint recovery commits every 25 frames. After the whole suite completes, its generated feature/replay caches under `data/benchmark_suite/<run>` are removed by default. Set `retain_completed_state: true` to retain them. Metrics, logs, official prediction files and videos remain; existing datasets, training checkpoints and older experiment folders are preserved.

`python benchmark.py suite --report-only` reads saved evidence without inference. It does not rewrite HTML unless a new best is discovered. Stage logs are under `metrics/benchmark_suite/<run>/logs/`. Detach tmux with Ctrl+B, D; reattach with `tmux attach -t reid-bench`.

## Coverage limits

GT-box recovery isolates appearance/association. Joint recovery uses crop soft-mask embeddings and transfers specialist thresholds unchanged; it is not calibrated full-frame or cross-camera recovery. Three forced target returns are diagnostic only. KITTI timing includes decode/model/association/RLE, while reference timing is crop-encoder-only; fresh detector-to-annotated-MP4 timing remains open.

Joint VRU/UAV-VeID/weather retrieval and joint CityFlow remain unmeasured. Published recipes differ and are labeled references. Complete the current joint assessment before RF-DETR Seg, then test Cutie/McByte++ separately.
