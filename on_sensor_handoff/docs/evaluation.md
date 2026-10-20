# Evaluation protocol

## Dataset roles

| Source | Proposed role | Limitation to preserve in claims |
| --- | --- | --- |
| [CO3D](https://ai.meta.com/datasets/co3d-downloads/) | Multiview common-object identity training or pretraining | Moving one camera around an object is not a two-camera handoff. |
| [TAO](https://taodataset.org/) or [GMOT-40](https://spritea.github.io/GMOT40/) | Diverse category tracks, within-view temporal variation, and proposal evaluation | Track IDs are not automatically cross-camera identity labels. |
| [BuckTales](https://papers.neurips.cc/paper_files/paper/2024/file/95286b5d4cd5b7953bd2bbe717300fe0-Paper-Datasets_and_Benchmarks_Track.pdf) | Hard cross-drone test with lookalike animals | Identity mappings are within video pairs, not verified across recording days. |
| Locally recorded paired AI Cameras | Final cross-camera and actual-sensor validation on varied physical objects | Define capture sessions and verify cross-camera identities manually. |

Additional person/vehicle/product identity datasets may broaden training, but
record their licenses, camera labels, and identity scopes before mixing them.
Do not use category labels or adjacent frames as substitutes for physical
instance labels. Freeze validation and test data before tuning thresholds.

The first [AICity22 vehicle-only smoke test](aicity_smoke_v1.md) freezes an
actual local cross-camera split and measures appearance baselines using
ground-truth candidate boxes. It checks the pipeline; it supplies no evidence
for unseen object categories or live on-sensor handoff.

## Splits

1. **Identity-disjoint:** no physical identity appears in both training and
   evaluation within a dataset. Keep full tracks together.
2. **Recording-disjoint:** keep all observations from a paired recording or
   capture session in one split, avoiding background and time leakage.
3. **Category-held-out:** reserve whole object categories for the generality
   claim. Report seen and unseen categories separately.
4. **Cross-camera query/gallery:** choose the reference from camera A and
   candidates from camera B. Include scenes where the referenced object is
   absent. For one-camera multiview sources, label the result as multiview
   retrieval rather than cross-camera handoff.

For BuckTales, split by video pair or recording segment. Its stated 730 tracks
and hundreds of thousands of frames are not that many independent identities
or capture conditions. Report results across multiple held-out groups.

## Baselines and ablations

| System | Purpose |
| --- | --- |
| Time/position or track-continuity matching alone | Tests whether appearance adds information. |
| `Pxl-loc-tak` 140-value Pi-computed fingerprint | Existing inexpensive embedded baseline. |
| `seg_tracking` host appearance encoder and handoff | Existing higher-compute application baseline. |
| Small FP32 encoder on host | Separates representation quality from sensor conversion. |
| Same encoder, quantized on IMX500 with selected ROI | Measures on-sensor identity quality and throughput. |
| Joint detector + embedding on IMX500 | Tests whether one pass improves total handoff performance. |

Hold input data, candidate set, matching policy, and bandwidth budget fixed
where possible. Compare detector recall separately when candidate sets differ.
The first optional ablations are backbone, embedding dimension, and temporal
fusion. Add mask supervision only if suitable masks or audited pseudo-masks are
available; boxes alone do not supervise a segmentation head.

## Measurements

| Level | Required measurements |
| --- | --- |
| Representation | Cross-view Rank-1/mAP; verification ROC or TAR at fixed FAR; no-match false accepts; performance by seen/unseen category, object size, and viewpoint. |
| Handoff | Correct, false, missed, and rejected handoffs; time from reference publication to match; identity switches after acquisition. |
| Sensor | Converter memory use, RPK size, DNN and DSP runtime, actual embeddings or candidates per second, dropped/missing inference outputs, quantization-induced ranking changes. |
| System | Host CPU/RAM, transmitted bytes per handoff, candidate count, ROI scheduling overhead, live end-to-end latency. |

Do not equate a model's 10 inferences/s with 10 updates/s for every target.
Report the number of targets competing for the sensor and the rate each target
actually receives. A replay experiment supports algorithm comparisons; a live
two-camera experiment supports the end-to-end deployment claim.

## Capstone and thesis completion bars

**Capstone:** a reproducible two-camera demonstration that transfers one
user-selected object across views, with at least one learned IMX500 embedding
model, a no-match case, and measured latency/false matches against the current
handcrafted fingerprint. Show results on several distinct object categories.

**Thesis:** a frozen identity- and category-held-out evaluation; quantitative
comparison of selected-ROI and joint-model paths (or a documented joint-model
feasibility boundary); repeated hardware measurements; analysis of failure by
object size, category, and viewpoint; and reproducible code, splits, model
artifacts, and result tables. A thesis claim should specify the operating
conditions and improvement, rather than promise reliable matching of every
physical object. Then make the single-pass detection + embedding model your principal research attempt. Compare them on held-out object categories using handoff success, false matches, missed objects, latency, and sensor memory. That gives you a complete thesis even if the one-pass model exposes a hard resolution or memory limit—and a stronger result if it succeeds. 
