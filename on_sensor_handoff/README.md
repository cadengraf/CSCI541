# On-sensor object identity handoff

## Research question

Given an example of a physical object in camera A, can camera B find the same
instance and reject a match when it is absent, while the Sony IMX500 performs
the learned identity inference? The target category may be absent from training.
"Category-general" here means tested on held-out categories with visible
instance cues; it does not promise that visually identical or unresolved objects
can be distinguished by appearance alone.

This project has two implementation paths:

1. **Selected-ROI embedding (first implementation path):** a generic proposal/tracking
   system supplies candidate regions; a small IMX500 model produces one
   embedding for the selected region per inference. The host schedules
   candidates, matches embeddings, and makes a match-or-abstain decision.
2. **Joint detection and embedding (research model):** one IMX500 network uses a
   shared backbone to output object locations and an identity embedding for
   each candidate in the same inference. The host handles temporal association
   and cross-camera matching.

The first path is a full experimental baseline and a viable capstone. The
second path tests whether one sensor model improves handoff throughput or
accuracy under the same resource budget. A controlled finding that the joint
model loses small objects or cannot fit also defines the useful limit of the
sensor, provided the first path and measurements are complete.

## Existing systems to build from

| Existing project | Capability to reuse | Boundary for this project |
| --- | --- | --- |
| `seg_tracking` | Prompt-based FastSAM selection, CUTIE mask tracking, appearance matching, publisher/subscriber handoff, and a compact registry protocol | Its current ONNX appearance encoders run on the host; the exported pretrained features have no project-trained cross-view ReID head. Use it as the application and stronger-model reference. |
| `Pxl-loc-tak` | IMX500 RPK loading, C++ tensor post-processing, detection stabilization, camera streaming, and UDP metadata | Its current six-class detector does not discover arbitrary objects; its 140-value appearance descriptor is computed from YUV video on the Pi. Use it as the sensor runtime and inexpensive descriptor baseline. |

Keep both projects external during initial research. Adapters can take their
locations as configuration instead of copying either codebase into this tree.
The current IMX500 post-processor expects four detection tensors, and its ROI
setup covers the full sensor image. An embedding RPK requires a different tensor
decoder and selected-ROI control; a joint RPK requires a decoder for detections
**and** per-instance features.

## Work order and decision gates

| Stage | Deliverable | Gate before proceeding |
| --- | --- | --- |
| 0. Data and reference | Versioned identity manifest, leakage-safe splits, and frozen `seg_tracking`/handcrafted-descriptor baselines | The task and splits are sound, and baseline errors are quantified on held-out recordings and categories. |
| 1. Sensor embedding | Small trained encoder, int8 export, IMX500 tensor reader, ROI benchmark | It compiles, fits memory, preserves useful identity ranking after quantization, and reaches the measured throughput target. |
| 2. Two-camera handoff | Camera A publishes a model-versioned reference; camera B samples candidates and either matches or abstains | False handoffs, missed handoffs, latency, and bytes are measured on replay and live cameras. |
| 3. Joint model | One network emits locations and embeddings; identical evaluation to stage 2 | Compare the full tradeoff with the selected-ROI system, including detector recall on small objects. |

The provisional target is about **10 total embedding inferences per second** for
the selected-ROI path. It is not 10 FPS for every object when several candidates
share the sensor. The joint path is measured in useful *objects embedded per
second* as well as frames per second.

## Documents

- [Pipeline and component contracts](docs/pipeline.md)
- [Datasets, comparisons, and success criteria](docs/evaluation.md)
- [Frozen AICity22 vehicle smoke test and measured baselines](docs/aicity_smoke_v1.md)
- [Train, export, capture sensor output, and compare](docs/model_to_sensor.md)

When implementation grows, keep importers, training, evaluation, IMX500 export,
device runtime, and handoff adapters in separate modules under this directory.
The two existing applications remain external references until their interfaces
are stable. This directory can then become the root of a standalone repository
without carrying course homework or machine-specific paths with it.

## First executable slice

1. Inventory available IMX500 hardware and the tested RPK/compiler versions.
2. Build the data manifest from a small multiview identity subset. The
   [example manifest](examples/observations.jsonl) and
   [validator](tools/validate_manifest.py) establish the first data contract:
   `python on_sensor_handoff/tools/validate_manifest.py on_sensor_handoff/examples/observations.jsonl`.
   Freeze train/validation/test groups before training.
3. Run a pretrained `seg_tracking` appearance encoder and the Pi's handcrafted
   descriptor on the same held-out pairs. This establishes whether the data and
   matching task are viable before sensor model work.
4. Train one small projection/encoder against cross-view positives and
   same-category hard negatives. The [model-to-sensor instructions](docs/model_to_sensor.md)
   and executable tools cover training data, FP32 export, MCT conversion,
   RPK packaging, raw sensor capture, and the same match-or-abstain comparison.

Generated datasets, models, and run outputs belong in the ignored `data/`,
`artifacts/`, and `runs/` directories. Source, split definitions, provenance,
configuration, and aggregate results should remain reproducible in Git.

## Related work and hardware reference

- [FairMOT](https://arxiv.org/abs/2004.01888) and
  [CrossMOT](https://arxiv.org/abs/2302.07676) establish that a joint
  detector/embedding network is prior art.
- [Object-Generalized Re-Identification](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Object-Generalized_Re-Identification_A_Step_Towards_Universal_Instance_Perception_CVPR_2026_paper.pdf)
  motivates explicit evaluation on unseen categories.
- [Raspberry Pi AI Camera documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
  describes ROI control, output tensors, and model packaging. Its
  [product brief](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf)
  lists the maximum model input and shared on-chip memory; converter reports
  and live measurements are authoritative for each candidate network.
