# Target: full-frame vehicle detection, instance masks, and ReID

## The research question

Can one lightweight network detect vehicles, output an identity embedding for
each candidate, and produce an **instance mask and mask-guided 128D embedding
for the selected vehicle or a short list of candidates**, while retaining competitive
cross-view identity accuracy at lower end-to-end latency than a larger joint
model or a detector/segmenter plus separate ReID model?

This is the intended architecture. The new `utils/full_frame/model.py` is a
working first full-frame baseline: it shares a COCO-pretrained ResNet-50-FPN image pass
across detection, per-instance masks, and a mask-pooled 128D ReID head. Its
KITTI training entry point is `python train.py full-frame`. It still needs
training, measured full-frame evaluation, and the selective mask optimization
described below. The older `JointNet` in `utils/core/model.py` is a **crop-only
predecessor**: it predicts one binary
foreground mask and one embedding for an already selected crop. It cannot
detect vehicles in a full frame or produce distinct masks for adjacent cars.
The current VRU/VRAI metrics measure crop retrieval only.

## Proposed one-network design

```text
640 × 640 drone frame
       │
       ▼
light shared backbone + multiscale features
       │
       ├── vehicle detection head ──► boxes and scores for candidates
       └── ReID feature head ───────► coarse embedding per candidate
                                         │
                              selected / top-K candidates
                                         │
                       per-object mask head (only selected RoIs)
                                         │
                       mask-guided pooling on shared features
                                         │
                                         ▼
                         refined 128D embedding + mask
```

The network must accept a complete frame and return candidate `boxes`,
`scores`, and coarse `embeddings`, then one or a few selected `masks` and
mask-guided `embeddings`. Identity matching between frames/drones still uses a
small association algorithm outside the neural network; that algorithm
compares embeddings and motion and preserves a selected target. The selected
masks must be **instance masks**, not one semantic vehicle map.

## Selective mask inference

Run the detection and coarse ReID heads for every candidate, since they are
needed to find the target. Then choose the likely target from motion plus
coarse appearance, and apply the mask head to **one RoI**. If the target has
been lost or appears in another drone's view, mask the top 2–5 candidates
and rerank them using refined mask-guided embeddings. The first selected
frame can use the user's clicked box as the target hint.

Train the mask head on **all labeled vehicle instances** so it generalizes;
the selective behavior is an inference optimization. A per-RoI mask head
(Mask R-CNN/CenterMask style) saves work when fewer RoIs are masked. A
YOLACT-style prototype mask head still computes shared full-frame mask
prototypes, so selecting fewer vehicles mostly saves mask assembly and
postprocessing, not the whole mask-head cost. One model can contain all the
heads, while the lightweight selection/association logic remains outside it.
Use the same 128D projection for coarse box-pooled and refined mask-pooled
features so both can be compared in one embedding space. Train both paths;
otherwise the initial candidate ranking could be poor even if the refined
embedding is strong. Backbone and detection cost still applies to every frame,
so the speedup depends on how expensive the per-RoI mask head is.

Selective output changes the task definition: the system segments the
**selected target**, not every vehicle in the frame. For an apples-to-apples
MOTS benchmark, run the mask head on all detections; for the deployed
single-target mode, report latency with one mask and with top-K fallback.

The first implemented baseline deliberately prioritizes a working architecture:
COCO-pretrained Mask R-CNN ResNet-50-FPN v2 provides the full-frame detector
and instance-mask decoder, and a mask-pooled 128D ReID head shares its FPN
features. The COCO car and background predictors initialize the two-class
heads. The ReID head pools under the model's **predicted mask** during both
training and inference, so a ground-truth-mask training shortcut does not
create a teacher/student mismatch. It computes a mask and embedding for each
detected vehicle. The next
functional milestones are real-data training, validation, a selected-target
video, and a matched separate-model control. Coarse box embeddings and
selective mask computation come after the all-instance version works.

## Teacher first, compact student later

The ResNet-50-FPN model is the **teacher candidate**, not the intended final
edge model. Its detector and mask head start from COCO weights; the ReID head
still needs identity supervision. KITTI MOTS provides within-video vehicle
tracks for the initial joint training. Cross-drone ReID requires an additional
stage with genuine cross-camera vehicle identities such as VRAI; KITTI track
IDs alone do not establish cross-drone identity accuracy.

Before distillation, measure the trained teacher's box AP, mask AP, ReID
Rank-1/mAP, HOTA/AssA, and end-to-end latency on fixed held-out splits. Save
the exact checkpoint and evaluation protocol. A smaller student can then
learn from both the real labels and the teacher's outputs: matched boxes and
scores for detection, soft instance masks for segmentation, and normalized
128D embeddings or pairwise similarities for ReID. Compare the distilled
student with the same small architecture trained without distillation, using
the same data and training budget. This isolates whether the teacher helped.
No teacher predictions from the final test set should enter student training.

This staged plan follows established object-detection and vehicle-ReID
distillation research. See [General Instance Distillation](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_General_Instance_Distillation_for_Object_Detection_CVPR_2021_paper.pdf)
and [Viewpoint Robust Knowledge Distillation for Vehicle ReID](https://link.springer.com/article/10.1186/s13634-021-00767-x).

Once that baseline is measured, compare smaller backbones or decoders such
as ShuffleNetV2-FPN, CenterMask2-Lite, or YOLACT-style prototypes. Keep the
data split, image resolution, and evaluation protocol fixed. The current
`image_size: 640` setting preserves aspect ratio and caps the longest side;
fixed 640 × 640 deployment and RPK conversion are separate checks. The
crop model checkpoint is not a trained full-frame detector.

## Training labels already present and missing

| Source | Full frames | Boxes | Instance masks | Physical IDs | Role |
| --- | --- | --- | --- | --- | --- |
| iSAID annotation | Matching train/val images currently absent from checked local paths | Yes | Yes | No | Aerial detection and masks once the matching images are obtained |
| VisDrone MOT | Yes | Yes | No | Track IDs within a video | Aerial detection and within-video embedding training |
| VRAI | Crops | Crop extent only | No | Cross-camera vehicle IDs | Cross-view embedding training and retrieval evaluation |
| VRU | Crops | Crop extent only | No | Vehicle IDs; camera IDs not released in current split | Aerial embedding training and crop retrieval evaluation |
| KITTI MOTS | Not yet in this repo | Yes | Yes | Track IDs within a video | Joint box/mask/embedding pretraining and standard MOTS evaluation |

The local iSAID `images-2` and `images-3` directories were checked against
`train/Annotations/iSAID_train.json` and `val/Annotations/iSAID_val.json`:
neither contains a filename from those annotation splits. Confirm the correct
iSAID train/val images before starting full-frame mask training. Existing
`data/aerial/vehicle_masks/` contains object crops, not the original scenes.

Train with whichever labels each source has: detection and mask losses on
iSAID; detection and identity losses on VisDrone; identity loss on VRAI/VRU
crops; all three losses on KITTI MOTS. Keep identities scoped to their source
and video. Do not call a within-video track ID a cross-drone identity.
Mask-guided embedding learning is strongest when masks and identity labels
co-occur, which makes KITTI MOTS useful despite its street-level viewpoint.

## Evaluation that tests the actual claim

| Claim | Metric | Required comparison |
| --- | --- | --- |
| Finds aerial vehicles | Box AP / AP50 and recall by object size | Same held-out aerial scenes and input size |
| Separates adjacent vehicles | Instance-mask AP / AP50 and per-instance IoU | Held-out iSAID scenes; include crowded examples |
| Preserves identity | Rank-1 and mAP on a fresh cross-camera identity split | Same proposals/query/gallery for all ReID baselines |
| Tracks a selected vehicle | HOTA, AssA, IDF1, and identity switches | Full-frame video with identity ground truth |
| Is faster | End-to-end milliseconds/frame and peak memory | Same hardware, 640 × 640, including mask assembly and association |

The old crop-only VRU/VRAI rankings remain a **ReID branch baseline**. They
are not a ranking of full-frame MOTS systems. Benchmark against MOTSNet and
MG-MOTS on a shared full-frame protocol, plus a separate
detector/segmenter + ReID pipeline. For a thesis claim, show a reproducible
speed/accuracy improvement and an ablation proving what mask guidance adds.
Published KITTI MOTS scores do not directly compare with aerial VRAI scores.

An initial project success criterion is **at least 25% lower median end-to-end
latency than a reproduced MOTSNet baseline on the same hardware**, while
keeping mask AP and tracking HOTA within two percentage points, or improving
either accuracy metric at similar latency. The cross-camera ReID mAP should
also meet or exceed a matched detector/segmenter plus ReID baseline. This is
a proposed project bar, not a published threshold or a claim already achieved.
Report the exact
input size, proposal limit, postprocessing, and hardware for every method.

## Closest papers and code

- [MOTSNet (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Porzi_Learning_Multi-Object_Tracking_and_Segmentation_From_Automatic_Annotations_CVPR_2020_paper.pdf): one network with detection, masks, mask pooling, and identity vectors.
- [MG-MOTS (2023)](https://ksp.etri.re.kr/ksp/article/file/67007.pdf): lighter CenterMask-based detection, masks, and mask-guided ReID in one network; same-video person tracking, not aerial cross-drone vehicle matching.
- [CenterMask2-Lite code](https://github.com/youngwanLEE/centermask2): available lightweight detection and instance-segmentation starting point. Its published speed is measured on a Titan Xp, not this project hardware.
- [YOLACT (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.pdf): fast one-stage prototype masks; no ReID head in the original method.
- [KITTI MOTS benchmark](https://www.cvlibs.net/datasets/kitti/eval_mots.php): full-frame masks and track IDs for cars, useful for unified training/evaluation but not aerial imagery.

The first coding milestone now exists in `utils/full_frame/`: a full-frame
forward pass produces distinct vehicle boxes, instance masks, and embeddings
from **one shared network**. A synthetic training/inference smoke test runs via
`python train.py full-frame-smoke`; the KITTI MOTS loader was checked against
real compressed car-mask annotations. Once the image archive finishes, prepare
it with `python prepare.py kitti`, inspect a real batch, and train with
`python train.py full-frame`. Next implement held-out box/mask AP and KITTI
tracking evaluation, then compare mask-guided versus box/global pooling and
selective versus all-instance masking under the same trained weights.
