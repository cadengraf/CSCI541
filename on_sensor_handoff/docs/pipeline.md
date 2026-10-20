# Pipeline and component contracts

## End-to-end data flow

```text
Camera A: select physical object -> local track -> reference embedding
                                                 |
                       model ID + target ID + timestamp + compact embedding
                                                 |
Camera B: scene -> candidate proposals/tracks -> geometry/time gate
                  -> sensor embedding budget -> compare across observations
                  -> match one candidate or abstain -> local track
```

Candidate proposal is distinct from identity matching. The initial system can
use the generic FastSAM/CUTIE path in `seg_tracking` for proposals and local
tracking. The IMX500 performs learned embedding inference on selected regions.
This supports an **on-sensor ReID** claim while stating clearly which work is
done on the companion computer. The joint model later moves proposals and
embeddings into the same IMX500 inference. The initial selected-ROI runtime
loads an embedding RPK in place of the existing `Pxl-loc-tak` detection RPK;
its candidate proposals must therefore come from the host or another source.

## Component boundaries

| Component | Input | Output | Initial owner |
| --- | --- | --- | --- |
| Reference selection | Frame and point/box prompt | Object mask, local track ID | `seg_tracking` adapter |
| Candidate discovery | Camera B frame | Candidate boxes/masks and local IDs | Generic host proposal adapter |
| ROI controller | Candidate box in full sensor coordinates | One inference region and associated frame/track ID | IMX500 runtime adapter |
| Identity encoder | Selected image region | L2-normalized vector and model version | IMX500 RPK |
| Matcher | Reference vector, candidate vectors, optional motion/position | Score per candidate; match or abstain | Host |
| Transport | Target/model/camera IDs, timestamp, vector, quality | Versioned packet or registry state | `seg_tracking` registry adapter |

Both cameras must use compatible embedding models and identical preprocessing.
Never compare vectors from different model versions as though they share a
space. Log the ROI applied to each inference and the frame to which its output
belongs; an asynchronously updated ROI can otherwise label the wrong crop.

## Data contract

Use a manifest whose rows describe **observations**, not images randomly
assigned to splits. The minimum fields are:

```json
{
  "schema_version": 1,
  "dataset": "example",
  "capture_group": "session-01",
  "camera_id": "camera-a",
  "track_id": "track-17",
  "identity_id": "session-01/object-03",
  "category": "backpack",
  "timestamp_us": 1234567,
  "image_path": "data/example/session-01/a/frame-0001.jpg",
  "bbox_xywh": [100, 80, 45, 60]
}
```

`identity_id` denotes the same physical instance across cameras *only where
the source actually supplies that mapping*. Prefix it with the capture group
when identity is local to a recording. `capture_group` is the unit that stays
together during a recording-disjoint split. Add optional mask path, viewpoint,
sensor name, and visibility metadata when available. Keep dataset licenses and
original annotation provenance with the import adapter. Run the
[`validate_manifest.py`](../tools/validate_manifest.py) tool after each import;
it checks field types, bounding boxes, duplicate observations, and whether a
camera-local track changes identity or category. File-existence checks are
optional so a manifest can be reviewed before large assets are staged.

Training pairs come from the same `identity_id` across distinct observations;
prefer distinct cameras or substantial viewpoint changes. Hard negatives are
different identities of the same category in the same scene. Subsample nearby
video frames so long tracks do not overwhelm the number of independent objects.

## Model paths

### A. Selected-ROI encoder

```text
input ROI -> compact backbone -> pooled features -> raw 64-D projection
          -> host L2 normalization and cosine matching
```

The first executable model trains a projection on vehicle identities while
keeping the pretrained backbone frozen. Later training needs physical identity
labels from several categories, with whole categories held out for testing.
Compare the pretrained feature, frozen-backbone projection, and limited
backbone fine-tuning before choosing a candidate for conversion. Evaluate
FP32, quantized export, and actual IMX500 outputs on matched observations;
the sensor's input ISP and host image crop need a separate parity check.
Record the converter memory report and live DNN/DSP timing.

The current `Pxl-loc-tak` bridge has a four-output detection decoder. Introduce
an explicit encoder-output decoder instead of interpreting a vector as NMS
boxes. Extend its full-frame ROI setter to take a selected box, and attach
camera timestamp, ROI, and candidate ID to each output. Keep its current
handcrafted descriptor as a baseline.

### B. Joint detection and embedding model

```text
full inference region -> shared compact backbone
                      -> objectness/box head
                      -> per-instance embedding head
                      -> K candidate boxes + K vectors
```

Train with localization and cross-view identity losses. Because the sensor
input is much smaller than full-resolution video, first measure candidate
pixel sizes and detection recall at the actual model input resolution. Define
a fixed, converter-compatible output layout and measure its tensor transfer
cost. A dense feature map sampled on the Pi and a fixed-K box/vector output
are implementation candidates; choose after testing converter support and
memory. The host still manages track state, matching, and abstention.

## Runtime decision policy

Start with three increasingly capable policies using the same embeddings:

1. Rank candidates by detector confidence or track age and visit them in turn.
2. Gate by compatible time/position and visit only plausible candidates.
3. Prioritize candidates whose current evidence is ambiguous; stop when a
   calibrated threshold and margin support a unique match, or abstain.

Use the first two as controls for any proposed scheduling method. Reserve a
strict compute budget in *total sensor inferences per second* and log every
candidate skipped or revisited. The joint model must be compared against the
same handoff task and sensor/host budget, not just frame FPS.
