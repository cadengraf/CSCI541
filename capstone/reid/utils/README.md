# Implementation map

Four small root scripts dispatch lazily through `cli/`. Calling them with no command or `--help` starts no work.

| Folder | Responsibility |
|---|---|
| `cli/` | Current command routing and configuration compatibility; `legacy/` retains old command dispatch. |
| `teacher/` | Shared Co-DINO / RefineMask / 256D ReID model and joint training. |
| `distillation/` | Frozen specialist targets and joint continuation/control. |
| `reid_teacher/` | Independent TransReID construction, training and frozen loading. |
| `vehicle_reid/` | Video pipeline, quality gates, identity memory, local/global association and existing benchmark helpers. |
| `experiments/aerial_training/` | Audited mixed VRAI/VRU identity data and matched training arms. |
| `experiments/aerial_benchmark/`, `aerial_transfer/`, `vehicle_reference/`, `vehicle_recovery/`, `cityflow/` | Current retrieval and video/cross-camera evaluation protocols. |
| `experiments/reid_balance/`, `roi_reid/`, `crop_training/`, `vrai_adaptation/` | Joint-model ablations; shared training dependencies are intentionally retained. |
| `experiments/appearance/`, `appearance_clip/` | Early appearance pilots and shared losses/augmentation still imported by current aerial training. |
| `reporting/` | Short dashboard plus detailed HTML appendix, generated only from existing evidence. |
| `data_sources/` | Dataset/manifest preparation. |
| `full_frame/` | Reusable local tracker, KITTI protocol and evaluator, plus retained legacy full-frame prototype. |
| `benchmarking/` | Native reference adapters, retrieval metrics and retained old report utilities. |
| `core/`, `crossdrone/`, `visualization/` | Legacy crop model / broad-object workflows and shared helpers used by existing adapters. |
| `diagnostics/`, `tracking/` | Existing joint diagnostics and earlier tracking integrations. |
| `vendor/` | Local pinned upstream checkouts; ignored by Git. |

Model/training/association import paths and signed source modules stay stable so existing checkpoints and the paused CityFlow run remain resumable. Historical command and model support is retained behind the smaller current interface, rather than silently breaking saved state.

See [architecture](../docs/architecture.md), [guide index](../docs/README.md), and [archived experiments](../docs/archive/README.md). No long run is launched without the user's explicit instruction.
