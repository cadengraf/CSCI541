# Historical experiments and prototypes

These guides are preserved to explain existing checkpoints and results. They are not the active roadmap. The original Python import paths and YAML configs are intentionally retained for signed checkpoints and paused-run compatibility.

- [Crop-only predecessor architecture](crop_model_architecture.md)
- [Old full-frame / Mask R-CNN prototype](full_frame_prototype.md)
- [Broader aerial dataset notes](aerial_data_notes.md)
- [Joint crop geometry](crop_training.md), [crop diagnostic](crop_diagnostics.md)
- [Joint versus ReID-only loss balance](reid_balance.md)
- [Shared-ROI spatial adapter](roi_reid.md), [earlier ReID diagnostics](reid_diagnostics.md)
- [Joint VRAI-only adaptation](vrai_adaptation.md)
- [Early specialist appearance pilots](appearance_training.md)
- [Local/global tracklet design](vehicle_tracklets.md)
- [Recovery budget](recovery_budget.md), [temporal evidence ablation](tracklet_evidence.md)
- [Paired original/degradation encoder benchmark](vehicle_encoder_benchmark.md)
- [Detector candidate notes](detector_frontends.md)
- [Earlier tracker / CUTIE experiments](tracking.md)
- [Legacy KITTI MOTS protocol](kitti_mots_benchmark.md)

Access historical command help with `python train.py legacy --help`, `python benchmark.py legacy --help`, or the corresponding demo/prepare entry point. Existing explicit command spellings continue to work. No-mode entry points now show help; the original default workflow is available only through the explicit `legacy default` command.

The unrelated UAVid helper checkout and the superseded standalone 48px appearance-diagnostic launcher were removed after creating a recovery backup. Data, checkpoints, existing result tables and videos were not removed.
