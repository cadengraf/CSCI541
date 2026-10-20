# First trained identity model and AI Camera comparison

This stage keeps candidate boxes known so the experiment can isolate identity
inference. It trains a 64-value projection on a pretrained EfficientNet-B0
backbone, exports a fixed 160×160 model, quantizes it for IMX500, and records
**actual sensor output tensors**. Host-side L2 normalization and cosine matching
use the same decision policy as the [vehicle smoke test](aicity_smoke_v1.md).
The first training pool contains vehicles only; it establishes deployment
feasibility, not the claim that arbitrary categories can be re-identified.

## Model contract

| Boundary | Contract |
| --- | --- |
| Source observation | A manually selected ROI; camera A and B must use the same model version. |
| Network input | One RGB image, 160×160, float values in `[0,1]`. Crop is resized directly to a square. ImageNet mean and standard deviation are applied **inside** the model. |
| Network output | One raw 64-value embedding. No classifier, L2 normalization, detection, or identity decision in the sensor model. |
| Host | Check model hash and tensor length, L2-normalize, compare cosine scores, and accept or abstain at a threshold chosen on validation. |

The initial backbone is `timm` `efficientnet_b0.ra_in1k`, whose pretrained
weights are available in this workstation's cache. Only its linear projection
is trained for the first pass, preserving general visual features while the
local identity-labeled training pool is vehicle-only. Raspberry Pi's
[IMX500 model zoo](https://github.com/raspberrypi/imx500-models) includes an
EfficientNet-B0 model, which is useful feasibility evidence for the backbone;
the converter report decides whether **this** input size and head fit.

## 1. Prepare training data and fit the head

From this repository's root, set `SEG_ROOT` to the local `seg_tracking` checkout.
The builder uses all present development trials from AICity S01 and S03,
extracting one source/destination crop per cross-camera pair. S02 calibration
and S04 test identities are excluded. The original AICity license applies to
all generated crops. Run the existing manifest validator on the output.

```bash
python -m pip install -r on_sensor_handoff/requirements-train.txt
export SEG_ROOT=/home/megrad/Documents/GitHub/seg_tracking
export BENCHMARK_MANIFEST="$SEG_ROOT/aicity_handoff_benchmark/outputs/manifest.jsonl"
python -m on_sensor_handoff.tools.build_aicity_train \
  --source-manifest "$BENCHMARK_MANIFEST"
python -m on_sensor_handoff.tools.validate_manifest \
  on_sensor_handoff/data/aicity_train_v1/observations.jsonl --check-files
python -m on_sensor_handoff.tools.train_identity
python -m on_sensor_handoff.tools.export_identity
```

The checkpoint, FP32 ONNX, and input/output contract are written under ignored
`on_sensor_handoff/artifacts/`. The trainer uses cross-camera positive pairs
and different identities from the same scenario as in-batch negatives. It does
not use S02 or S04 for model selection. The export script checks PyTorch/ONNX
numerical parity and emits a `*.contract.json` file.

Before conversion, run the trained FP32 model on the **frozen S02/S04 trials**
alongside the previous baselines:

```bash
export PXL_ROOT=/home/megrad/Documents/GitHub/Pxl-loc-tak
python -m on_sensor_handoff.tools.run_aicity_baselines \
  --model "$SEG_ROOT/models/onnx/timm/reid_fastvit_t8.apple_in1k_256.onnx" \
  --fingerprint-source "$PXL_ROOT/coms_servers/opencv_contour_tracker/src/camera/lightweight_fingerprint.cpp" \
  --fingerprint-include "$PXL_ROOT/coms_servers/opencv_contour_tracker/include" \
  --identity-model on_sensor_handoff/artifacts/identity_b0_64_fp32.onnx \
  --output on_sensor_handoff/runs/aicity_smoke_with_identity.json
```

These are still oracle-box **host** results. AICity is an initial vehicle-only
checkpoint; hold out S04 while choosing model changes on training/validation.

## 2. Quantize and convert on the development computer

Use a separate Python environment for [Sony Edge-MDT](https://github.com/SonySemiconductorSolutions/aitrios-edge-mdt).
Its MCT package performs IMX500-specific post-training quantization, and
`imxconv-pt` creates `packerOut.zip`. The quantizer script uses real **train-only**
crops for calibration; random images or held-out test frames would invalidate
the comparison. The wrapper defaults to preserving the input tensor so it can
be inspected later if the Pi camera stack exposes it. If converter memory is too high, set
`IMX500_NO_INPUT_PERSISTENCY=1` and rerun the converter.

```bash
python -m pip install 'edge-mdt[pt]'
bash on_sensor_handoff/tools/convert_imx500.sh
```

The quantizer uses Sony's IMX500 target platform capability version `1.0` by
default. If your converter requires another supported version, pass
`--tpc-version` directly to `quantize_imx500.py` and then run `imxconv-pt`
on its output. Record the installed MCT and converter versions with the result.

Inspect the converter's memory report. A model that cannot be converted or
fitted is a measured feasibility result, not a valid RPK. Do not rename a
normal FP32 ONNX file as a quantized one. Sony's
[MCT export API](https://sonysemiconductorsolutions.github.io/mct-model-optimization/api/api_docs/modules/exporter.html)
describes the quantized ONNX format. Raspberry Pi's
[AI Camera deployment guide](https://www.raspberrypi.com/documentation/accessories/ai-camera.html#model-deployment)
documents `imxconv-pt`, the memory report, and Pi-side packaging.

Optional: pass `--quantized-model <MCT ONNX>` to the comparison runner to
measure post-quantization ranking on the held-out crops before using the Pi.
This needs `mct_quantizers` from the converter environment.

## 3. Package and capture on the Raspberry Pi

The packager runs **on a Raspberry Pi**. Transfer the ZIP by your normal method,
then run:

```bash
imx500-package -i packerOut.zip -o identity_rpk
```

This creates `identity_rpk/network.rpk`. Copy this repository's
`on_sensor_handoff/` directory and that RPK to drone 17 using your preferred
workflow. Stop any existing application that owns the camera before running
the capture script. You control these device actions.

For each validation or test scene, record one query ROI from camera A and each
candidate ROI from camera B. All records for one scene need the same
`--capture-group` and `--scene-id`. Assign the **same physical identity ID**
across cameras; use distinct IDs for different physical objects. `--roi` is
`x y width height` in full 4056×3040 IMX500 sensor coordinates, following the
ROI control used by `Pxl-loc-tak`. Use different physical identities and
capture groups for validation and test. Include scenes where the query is
absent from B's candidates.

If camera A and camera B are two positions of one moved sensor, label the
experiment **multiview**. A cross-camera claim requires separate cameras.

```bash
python -m on_sensor_handoff.tools.capture_imx500 \
  --model identity_rpk/network.rpk --camera-id camera-a \
  --capture-group val-session-01 --scene-id val-scene-01 \
  --identity-id val-session-01/object-1 --category backpack \
  --split validation --role query --roi 1200 800 600 700

python -m on_sensor_handoff.tools.capture_imx500 \
  --model identity_rpk/network.rpk --camera-id camera-b \
  --capture-group val-session-01 --scene-id val-scene-01 \
  --identity-id val-session-01/object-1 --category backpack \
  --split validation --role candidate --roi 900 700 550 650
```

Repeat the candidate command for every object in B's scene, changing the
identity and ROI. For a no-match scene, record the query on A but only other
physical identities on B. Capture test scenes with fresh object IDs, then
transfer the `capture_*.jsonl` files **and their referenced images** back to
the same relative directory in this repository. The script stores one raw
64-value sensor vector, model hash, sensor ROI, preview ROI, frame timestamp,
and DNN/DSP KPI per accepted frame. It discards warmup outputs after ROI setup.
The selected-ROI capture still uses manual candidates; no detection claim is
made at this stage.

For a sustained single-ROI rate sample, add `--count 50` to one capture run;
the printed rate uses output frame timestamps. This does not measure the cost
of switching between several candidates. Keep the resulting repeated frames
for diagnostics; the trial builder selects the earliest capture per identity.

## 4. Compare on the exact captured observations

```bash
python -m on_sensor_handoff.tools.build_live_trials \
  on_sensor_handoff/data/live_sensor_v1/capture_*.jsonl
python -m on_sensor_handoff.tools.validate_manifest \
  on_sensor_handoff/data/live_eval_v1/observations.jsonl --check-files
python -m on_sensor_handoff.tools.run_aicity_baselines \
  --data-dir on_sensor_handoff/data/live_eval_v1 \
  --model "$SEG_ROOT/models/onnx/timm/reid_fastvit_t8.apple_in1k_256.onnx" \
  --fingerprint-source "$PXL_ROOT/coms_servers/opencv_contour_tracker/src/camera/lightweight_fingerprint.cpp" \
  --fingerprint-include "$PXL_ROOT/coms_servers/opencv_contour_tracker/include" \
  --identity-model on_sensor_handoff/artifacts/identity_b0_64_fp32.onnx \
  --sensor-embeddings on_sensor_handoff/data/live_eval_v1/imx500_embeddings.jsonl \
  --protocol 'Live paired AI Camera captures; manually selected ROI; cosine top-1' \
  --output on_sensor_handoff/runs/live_sensor_v1.json
```

`build_live_trials` rejects mixed model versions and identities crossing
validation/test. The runner calibrates each method on validation absences and
reports held-out rank-1, correct handoffs, wrong identities, misses, and absent
false accepts. Also inspect the capture log's KPI and missing-output counts.
Live camera pixels, ISP scaling, ROI timing, JPEG compression, and sensor
quantization can all cause host/IMX500 differences; matching on the same saved
frame bounds the comparison but is not a bit-exact tensor test.

After this sensor gate, camera B needs a **category-general proposal and local
tracking source**. The first integration can connect `seg_tracking` proposals
and CUTIE tracks to the ROI scheduler while the AI Camera supplies the identity
vector. Benchmark proposal recall separately from identity ranking, then add
threshold/margin and multi-frame evidence for match-or-abstain decisions. A
joint detector+embedding sensor model is a later experiment once the selected
ROI path has measured accuracy and throughput.
