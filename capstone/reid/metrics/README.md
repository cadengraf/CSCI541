# Vehicle model metrics

**Open [index.html](index.html) first.** It shows the current results, the
thresholds to beat, and a suggested research target. [rankings.html](rankings.html)
shows every measured model.

[vehicle_scorecard.html](vehicle_scorecard.html) is a standalone checklist for
evaluating a future full-frame vehicle detection, segmentation, and ReID model.

These are **crop-only ReID baselines** for the full-frame system described in
[FULL_FRAME_MOTS.md](../FULL_FRAME_MOTS.md). They do not measure vehicle
detection, per-vehicle instance masks, or end-to-end tracking.

To refresh both pages without rerunning models:

```bash
python benchmark.py metrics
```

`python benchmark.py vru` and `python benchmark.py vrai` also refresh both
HTML pages after updating their CSV files.

`vru_benchmark.csv` and `vrai_local_benchmark.csv` are the latest measured results.
`python benchmark.py vru` and `python benchmark.py vrai` replace those files with
new results. Numeric scores in these benchmark files are fractions (0–1), while
the target files use percentages (0–100).

Fill the empty `next_joint_percent` columns in `reid_targets.csv` after a new
joint run. The first goal is to exceed the matched ReID-only model on both
Rank-1 and mAP. VRU uses one gallery image per identity, so its reported mAP
equals mean reciprocal rank. The local VRAI split uses cross-camera queries
and galleries but is a custom split of the official **training** images.
Both local test results have already been inspected during development;
use an untouched identity holdout for a final confirmatory claim.

`mask_targets.csv` records proposed project goals. The 70% IoU and 200-crop
thresholds are practical targets, not published benchmark records. Evaluate
on vehicle masks from scenes excluded from mask training. Report mask-guided
versus global pooling using the **same checkpoint**.

`model_comparison.csv` is a blank table for OSNet, TransReID, the matched
ShuffleNetV2 arms, and a separate segmenter-plus-ReID pipeline. Train the
competing ReID methods on the same identity data and score them on the same
query/gallery manifests. Record crop size and hardware before comparing
latency; include both networks in the separate pipeline cost.

`published_targets.csv` lists paper results for their **official protocols**.
Do not put local VRU or VRAI scores in its result columns. The VRAI paper's
test split is different from `vrai_local_benchmark.csv`, and the local VRU
gallery selection differs from the published VRU selection.

## Closest lightweight related work

- [MG-MOTS, Seong (2023)](https://ksp.etri.re.kr/ksp/article/file/67007.pdf):
  a CenterMask-based model with a slim VoVNet2 backbone, instance masks, and
  mask-guided ReID features. It is a direct architecture reference, evaluated
  for same-video tracking on KITTI MOTS rather than aerial cross-camera ReID.
- [MOTSNet, Porzi et al. (2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Porzi_Learning_Multi-Object_Tracking_and_Segmentation_From_Automatic_Annotations_CVPR_2020_paper.pdf):
  one network uses instance masks for embedding pooling. It is a close method
  reference but uses a larger instance segmentation architecture.
- [OSNet, Zhou et al. (2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.pdf):
  a compact ReID model (about 2.2 million parameters) without a mask head.
  It is a size-matched ReID baseline, not an existing joint mask/ReID model.
- [MobileInst, Zhang et al. (2024)](https://ojs.aaai.org/index.php/AAAI/article/view/28555):
  mobile-oriented instance segmentation and video tracking. Its mask and
  tracking metrics do not measure cross-camera ReID, but its instance decoder
  is useful if the project later adds full-frame instance selection.

The current ShuffleNetV2 model requires a vehicle crop as input. A full-frame
detector or proposal stage is additional compute and must be included in
end-to-end deployment measurements.
