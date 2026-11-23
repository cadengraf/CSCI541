# Controlled aerial video recovery

Run or resume from `capstone/reid`:

```bash
../.venv/bin/python benchmark.py vehicle-recovery
```

The launched GPU queue uses `tmux attach -t vehicle_recovery`; its `logs` window follows `/tmp/vehicle-recovery.log`. Settings are in `configs/vehicle_recovery.yaml`; the device is always automatic. The queue freezes inputs, extracts original-specialist embeddings, runs both association methods on every case, and renders one comparison video without another conversation turn. No training, threshold search, new model, or default deployment change is involved.

## What this gate tests

Three VisDrone2019-MOT validation videos provide 945 real frames and 20,738 valid vehicle-box observations. The original VRAI-trained TransReID specialist is fixed. Methods are the existing local reference and the previously implemented experimental temporal query-pool resolver, with its pre-existing score .82 / margin .08 and other settings unchanged. Both methods share the exact same local tracker settings, GT vehicle boxes, embeddings, and box-quality decisions. The resolver requires three reliable spaced observations spanning four frames before authorizing global recovery. Its thresholds were not selected on this suite.

Each video runs two independent scenarios from a fresh association state:

- `natural`: all valid observations, with return events defined by gaps in the source vehicle annotations.
- `target_blackout`: remove only the selected target's observations for 45 frames after 40 enrollment frames; other objects remain observable. This deliberately exceeds the 15-frame local lifetime. The underlying image pixels are not altered, and the video explicitly marks withheld observations. These events are synthetic observation outages, not natural disappearance.

The frozen targets are vehicle 107 in `uav0000117_02622_v`, vehicle 23 in `uav0000137_00458_v`, and vehicle 4 in `uav0000182_00000_v`. They were chosen from annotation coverage before evaluating embeddings or association outcomes. Target labels are benchmark prompts and evaluation evidence only; neither truth IDs, class IDs, nor occlusion labels enter association. Observations are ordered by coordinates, and the online interface receives only current boxes, embeddings, box-only reliability, and frame number.

This is an oracle-detection appearance/association test. GT boxes replace detector discovery; there are no predicted masks or mask propagation. Box-only reliability requires minimum side 24 pixels, two pixels from the border, and at most 50% area overlapped by another current box. All valid crops with minimum side 12 pixels remain observations even if unreliable. Ground-truth occlusion annotations are retained for inspection, never used as a trusted-quality oracle. The suite cannot demonstrate detector robustness, quantitative viewpoint invariance, cross-camera transfer, application restart persistence, or full end-to-end video tracking quality. Those are separate roadmap gates.

## Scoring

Evaluation keeps immutable physical ownership for each claimed identity and the first trusted identity for each physical vehicle. A later ID switch never silently becomes the new recovery target. A return has at least two absent annotation frames; long-gap results separately require at least 16. The selected target has a fixed original identity enrolled before its blackout; post-return enrollment cannot retroactively count as recognition.

Recovery means accepting the original identity within 30 frames of return. Delay is measured from the first return frame to the first accepted original-ID claim. Misses, returns with no prior trusted enrollment, and windows censored by another disappearance or sequence end are reported separately. Natural events from blackout replays are not pooled into the natural-scenario aggregate, since the scenarios reuse frames and identities.

Local reacquisitions and global recovery actions are audited independently against immutable identity ownership. Reports include correct, false and unscored claims, false claims to genuinely new objects, wrong-identity observations, selected-target false-match/missed-visible frames, and memory writes. Rejecting a crop is not a false reconnection. A selected-target match onto another vehicle is an error even if its mask or box looks plausible.

Missed events carry their observed quality/evidence/status history and a first diagnostic reason: no crop, no reliable crop, insufficient temporal evidence, ambiguity, a different identity binding, appearance/reference rejection, or baseline local fragmentation. This is a triage aid, not proof of causality. It identifies the next useful inspection without starting a broad architecture search.

The gate requires improved long-gap recovery, no increase in false-claim count, at least 20 scored reconnections, no unscored reconnections, and at most a 1% empirical false-reconnection rate. Even a diagnostic pass does not automatically promote a configuration: these three videos and repeated/synthetic events cannot establish a population reliability guarantee. A failed gate leaves milestone 3 open and points to one diagnosed failure for the next bounded change.

## Outputs and resuming

`metrics/vehicle_recovery/selection.pt` freezes the selected model, annotations, frames, case definitions, code and configuration before inference. `protocol.yaml` is its readable counterpart. A preflight-only selection archive precedes final scoring; no inference results use it. `queue.yaml` reports progress or failure. `summary.md`, `report.yaml`, and `gate.yaml` provide aggregate/per-case evidence and a promotion decision. `comparison.mp4` renders all six replays side by side, reference versus query pooling, with target correctness and explicit scenario labels. Display FPS is configured playback speed, not measured inference FPS.

`data/vehicle_recovery/features/` contains necessary resumable feature shards, committed every 25 frames. Both methods and scenarios reuse those embeddings. Replay checkpoints save complete algorithms, temporal evidence, evaluator ownership/events, and causal decisions every 25 frames. Rerunning the same command reuses committed state and completed cases; a lock rejects simultaneous queues. Changed signed inputs require new output/state paths. No crop copies, JSON or NPZ exports are generated.

Extraction timing covers image reads, preprocessing, embedding inference and transfers, separately from checkpoint writes. Association replay timing is measured separately. Detection, masks, rendering and application overhead are not included, so neither number is advertised as full-system FPS.

Four tests exercise original-ID versus latest-ID scoring, delayed natural recovery, controlled-versus-natural labeling, new-object false claims, unknown rejection, unenrolled/censored events, and exact interrupted/resumed behavior with the actual reference and temporal association implementations. The toy association test verifies a 45-frame gap defeats the local tracker and a consistent appearance can recover after the temporal evidence delay; actual video outcomes are reported independently.
