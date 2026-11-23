# Repository cleanup — 2026-11-18

- Four root entry points remain: `train.py`, `benchmark.py`, `demo.py`, `prepare.py`. Dispatch and config helpers now live in `utils/cli/`. Invoking an entry point without a command prints help only; existing explicit command names remain compatible.
- Removed the unused UAVid toolkit and standalone appearance-diagnostics pilot. Archived 17 historical guides under `docs/archive/`; shortened the README, architecture guide and implementation map.
- Kept datasets, weights, results, current modules, historical checkpoint dependencies and all model/training/association source unchanged. No training or benchmark jobs were launched.
- Fixed ignore rules: generated outputs stay local; required YAML configs, requirements and HTML source templates are visible to Git. File moves and newly visible source explain why the pending-change count increases. Nothing was staged or committed.

Validation: 73 unit tests passed; 211 Python sources parsed; 125 local documentation links resolved. Existing implementation hashes, CityFlow resume signatures and aerial training signatures match their saved versions.

Recovery snapshot: `/tmp/reid-cleanup-20261118-141347/`. It contains the original patch/status, `source-and-evidence.tar.gz`, removed packages and `cleanup-moves.json`. This temporary backup is outside Git; retain it elsewhere if needed before clearing `/tmp`.
