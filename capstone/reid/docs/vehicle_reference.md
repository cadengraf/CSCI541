# Bounded vehicle reference gate

Run or resume from `capstone/reid`:

```bash
../.venv/bin/python benchmark.py vehicle-reference
```

Settings: `configs/vehicle_reference.yaml`. The device is automatic. The detached GPU queue uses `tmux attach -t vehicle_reference`, with a `logs` window following `/tmp/vehicle-reference.log`. No training runs are started and no live deployment defaults are changed.

This completes roadmap milestone 1 with exactly three candidates: original VRAI TransReID, mixed VRAI/VRU AdaSP TransReID, and the existing clean-VRU author MSINet checkpoint. No additional architecture sweep is part of this gate. Their weights are checksummed; the MSINet source revision is verified. The joint model stays excluded from unseen VisDrone comparisons because its historical training used VisDrone.

All models use the exact 146 reference crops and 622 queries from the previously frozen chronological VisDrone protocol. There are 340 closed-set queries from 104 enrolled identities. Identical boxes, sequence-specific galleries, 60-frame minimum separation and unknown-withholding rules are reused. Models retain their native normalization: TransReID uses mean/std .5, MSINet uses ImageNet mean/std; both directly resize RGB crops to 256-square. The embedding dimensions are 3,840 and 768 respectively. Ground-truth boxes isolate appearance; this does not measure detector, masks, association, disappearance recovery, or cross-camera matching.

For a controlled rejection comparison, every candidate fits its own cosine/margin thresholds on the **same** 606 VRAI queries / 1,080 gallery crops from 313 audited source-fit identities. An identical empirical 1% false-unknown and wrong-known budget is applied. All source fits finish and are frozen before new target scoring. No VisDrone images or metrics fit or select a rejection threshold. Existing original/AdaSP target features are reused after provenance checks; no old extraction or training signatures are changed. These common-fit rejection results differ intentionally from the historical target report using different source calibration pools.

MSINet's author weights trained on VRU, so the VRU identities reserved by our later training program cannot serve as clean MSINet calibration. Only the common VRAI fit set is used here. Published weights and differing training data/budgets do not support an architecture-only causal comparison. The VisDrone protocol was previously observed; this is an added baseline on it, not a new blind test. Continued target-directed tuning would make it development data.

Cost measurements run each model in FP32 on the same machine at batch sizes 1 and 32. Five warmup passes precede 20 timed passes; CUDA is synchronized. The report includes median/p95 encoder latency, throughput, peak allocated memory, loaded parameter count and parameters in modules invoked during inference. Crop I/O, detector, mask propagation, association, rendering and model loading are excluded from these encoder timings. The workstation is shared, and these are not full online FPS measurements. Dataset extraction time includes image reads/cropping/normalization but excludes checkpoint storage.

`metrics/vehicle_reference/summary.md` and `summary.csv` are the single scorecard. Per-model YAML includes source calibration, known correct acceptance, known wrong acceptance, unknown false acceptance (natural/withheld separately), per-video and size/occlusion slices, training exposure and cost. `protocol.yaml`, `selection.pt`, and `thresholds.yaml` freeze the experiment. `gate.yaml` marks the reference comparison complete and records the next stage. `review.mp4` compares all three on up to 48 curated examples; it is not an accuracy sample. Necessary resumable features and decisions stay under `data/vehicle_reference/`; no crop copies or extra feature exports are written.

Both source and target extraction checkpoint every eight batches; rerunning reuses committed features and completed stages. A queue lock prevents duplicate runs. Changed code/config/inputs require new output/state paths. Tests cover both embedding dimensions, interrupted extraction, completed-stage reuse, invalid output dimensions and freezing source thresholds before target evaluation.

Stopping rule: close this reference gate after the three-model scorecard. Keep the original as the existing video integration reference and evaluate any replacement in a controlled video recovery suite at comparable false-reconnection rates. Do not start another appearance-training sweep or add EdgeTAM until the next gate identifies a concrete need.
