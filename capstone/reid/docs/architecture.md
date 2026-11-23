# Current architecture and intended joint model

## Intended deliverable

```text
Current frame → shared visual encoder / encoded feature pyramid
                         ├─ Co-DINO detection → boxes and scores
                         ├─ RefineMask → instance masks
                         └─ object ROI → dedicated ReID path → normalized 256D embedding
                                                   ↓
                         trusted references + bounded recent evidence
                                                   ↓
                         causal association → selected identity → annotated video
```

The current joint starting point uses ViT-L / Co-DINO / RefineMask. ReID receives encoder ROI features, not the output of the segmentation decoder. Mask-guided versus box/global pooling is an evaluated design choice. Sharing visual computation is useful only if identity quality remains adequate.

Implementation: `utils/teacher/`, `utils/distillation/`, and the matched ablations under `utils/experiments/`. See [joint training](teacher.md), [distillation](distill_joint.md), and [historical ROI feature clarification](archive/roi_reid.md).

## Existing accuracy reference

```text
Current frame → Co-DINO + RefineMask → 256-square RGB crops
                                           ↓
                         independent TransReID ViT-B/JPM (3840D)
                                           ↓
                         same identity-memory / association responsibilities
```

This reference runs through `demo.py vehicle`. Mixed VRAI/VRU experiments train the independent specialist; they do not update the joint ReID branch. The specialist can supervise the joint model during training. If removed at inference, its cost is training-only.

## Inference versus training

Online cost includes decode, visual encoders, detection, masks/propagation if used, ReID, gallery search, association, rendering and state writes. Count shared computation once. Provided-detection CityFlow runs measure encoder/association behavior and cannot establish live detector-inclusive FPS.

Identity classifiers, supervised losses and a removed distillation teacher are training-only. Keep immutable reference appearances beside gated recent evidence; uncertain crops must not overwrite trusted evidence. A no-confident-match result is valid. All identity inference uses current and past observations only.

The main open issue is the shared model's appearance quality. See [milestones.html](../milestones.html). Older crop-only and Mask R-CNN prototype descriptions are retained in [the archive](archive/README.md), not presented as the active architecture.
