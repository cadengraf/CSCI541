# Accuracy reference and controlled global recovery

The live default is now the measured accuracy reference:

**Co-DINO + RefineMask → independent TransReID specialist → existing motion/IoU/appearance tracker.**

```bash
# From capstone/reid; device/environment selection is automatic.
../.venv/bin/python demo.py vehicle
```

Settings: `configs/vehicle_reid.yaml`. Output: `data/videos/vehicle_reference/annotated.mp4`. The reference wrapper reproduces `VehicleTracker`'s assignments, motion, normalization, and appearance EMA exactly; trusted profile recording is separate and cannot alter tracking. Its expected score reference is the executed `legacy_specialist` result, 79.46 HOTA / 87.35 IDF1 on the fixed-detector local validation protocol.

This mode retains the existing local tracker's 15-frame age and its limitations. It does not advertise persistent global recovery after local expiration. Local IDs are local tracking evidence. The reference's original appearance EMA behavior is preserved for an exact baseline; separately recorded profile references/recent evidence only accept reliable observations.

## Experimental global identity recovery

The previous local/global resolver remains available, with its frozen settings and checkpoint support:

```bash
../.venv/bin/python demo.py vehicle --config configs/vehicle_reid_experimental.yaml
```

Explicitly loading a saved target profile requests cross-view appearance search, so the normal command automatically selects the experimental resolver and prints that fact:

```bash
../.venv/bin/python demo.py vehicle --profile data/videos/vehicle_reference/state.pt
```

The default profile output folder is separate (`data/videos/vehicle_reference_profile/`). A new run discards old spatial state and preserves only the selected target's trusted appearance evidence. This does not mean global recovery has passed the error budget.

## Development-only budget selection

```bash
../.venv/bin/python benchmark.py vehicle-budget
```

Configuration: `configs/recovery_budget.yaml`. The complete job resumes automatically. For tmux:

```bash
tmux new -s recovery_budget '../.venv/bin/python benchmark.py vehicle-budget > /tmp/recovery-budget.log 2>&1'
```

The workflow is:

1. Verify or prepare frozen Co-DINO predictions for training-partition sequences 0004, 0005, 0011, and 0020. They contain 1,821 frames and 44 natural GT visibility-return events. Predictions resume by frame. No model training or weight changes occur.
2. Extract specialist observations into resumable working checkpoints. Extraction never receives GT labels.
3. Replay the reference and 18 global-recovery configurations causally. Scores 0.70/0.78/0.82/0.86/0.90/0.94, margins 0.04/0.08/0.12, and three-hit confirmation were fixed before measuring development results. GT masks are used only after association, for scoring.
4. Select an empirical candidate maximizing correctly accepted global recoveries, subject to the configured error budget and evidence requirements.
5. Freeze the selected point before assessing it on all nine conventional validation sequences. The original experimental point and the exact reference are assessed as controls. Validation results never change the chosen point.

The detector was previously trained on these development sequences. This is association calibration, not independent detector validation. The nine validation sequences were also inspected in earlier architecture development; being excluded from this calibration does not make them an untouched final research test.

## What the budget means

The requested operating budget is:

**wrong scored accepted global recoveries / scored accepted global recoveries ≤ 1%.**

The default requires at least 20 scored global accepts and no unscored global claims before an empirical candidate is eligible. It maximizes useful correct recovery rather than rewarding rejection of everything. The YAML grid and support requirement are explicit experiment settings, not claims of universal calibration.

Metrics distinguish:

- **Global recovery:** an explicit accepted action by the persistent resolver. Its prior physical identity is compared with the current GT mask match.
- **Local reacquisition:** the existing local tracker reconnects within its retention window. These errors have their own counters and are excluded from the global-recovery error denominator.
- **Unscored global claim:** the evaluator lacks a sufficient prior/current mask-IoU identity label. It is never treated as a correct recovery, and the default eligibility rule rejects settings with such claims.
- **Natural return recovery:** the separate GT-visibility-gap metric, retained alongside action-specific scores. This prevents a high precision point from hiding poor coverage.

Repeated claims about one vehicle are correlated. The report therefore also gives a one-sided 95% upper bound on an identity having any false accepted recovery. This stricter identity-level diagnostic uses different units from per-claim error. It is not a distributional guarantee. An observed zero-error setting on a small sample cannot establish a 1% error bound. Confidence support is reported separately from empirical eligibility.

**No selected point automatically enables global recovery in the default pipeline.** Insufficient samples, unscored claims, or failed budgets leave the accuracy reference in place. The experiment can report that no supported point exists.

## Artifacts and reproducibility

- `metrics/vehicle_recovery_budget/selection.yaml`: selected development point, budget, support flags, and unchanged live-default mode.
- `development.csv` / `development.yaml`: all configurations, per-sequence counts, empirical error, and uncertainty.
- `report.yaml`: frozen validation assessment, separated local/global errors, and tracking-identity stream equivalence against the archived reference/original experimental predictions.
- Working checkpoints live under `data/vehicle_recovery_budget/`. They contain observation extraction, tracker state, pending recovery evidence, and evaluation state; there are no per-frame image, JSON, or NPZ exports.

Code/configuration/model/GT signatures protect resume. Use a new `state_root` for an intentional code or experiment change. Do not silently reuse completed legacy metrics as new action-specific audits: old checkpoints explicitly report that the separate audit is unavailable.

## Executed result

The initial 18-setting sweep found no supported point. At the original settings, development global actions were 0 correct / 6 wrong. The frozen validation audit found 4 correct / 2 wrong / 1 unscored global claims (33.3% error among the six scored claims). The default reference and original experimental identity streams exactly match all archived validation streams. See [executed result](../../metrics/vehicle_recovery_budget/summary.md).
