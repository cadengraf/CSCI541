# AICity22 cross-camera appearance smoke test, version 1

This is the first executable stage of the handoff project. It asks whether an
appearance descriptor can rank the correct vehicle from camera B's candidates
and reject a match when that vehicle is absent. The AICity22 Track 1 MTMC data
supplies cross-camera vehicle IDs. It covers **vehicles only** and cannot test
the desired category-general claim. All candidates come from ground-truth
destination boxes, so these numbers isolate **appearance matching with an
oracle gallery**; they are not an end-to-end handoff result or a guaranteed
upper bound for a different candidate pool.
The learned FastViT encoder runs on the host CPU here, not on the IMX500.

## Frozen data and protocol

- The tracked [split file](../splits/aicity_smoke_v1.json) selects 12 unique
  identities in each of S01 (train), S02 (validation), and S04 (test), one
  present and one absent trial per identity. The 12 IDs are evenly spaced
  through eligible sorted identity IDs; the first eligible positive trial per
  ID is paired with its benchmark absent trial. The source benchmark manifest
  SHA-256 is stored in the split file. Scenarios and identities do not cross
  splits. The AICity *dataset partition* is separate from this benchmark split;
  S04's source files happen to be under AICity's `train/` directory.
- The [importer](../tools/import_aicity.py) takes the last of the source trial's
  sampled boxes as the query, extracts its video frame, and uses **all**
  annotated destination vehicles in the benchmark destination frame as the
  gallery. It validates present/absent labels against destination GT.
  Extracted JPEG frames, the version-1 observation manifest, and the trial
  manifest live under ignored `on_sensor_handoff/data/aicity_smoke_v1/`.
- The [runner](../tools/run_aicity_baselines.py) uses the same candidates and
  cosine top-1 policy for both representations. The fingerprint invokes the
  original 140-value C++ function from `Pxl-loc-tak`, compiled with a small C
  wrapper. It receives planar YUV420 converted from decoded JPEG frames; live
  Pi camera YUV input can differ slightly. The pretrained FastViT T8 768-D
  ONNX model comes from `seg_tracking`; input resizing, crop, RGB conversion,
  and ImageNet normalization mirror its native service. Neither model has been
  trained on these selected identities as part of this project.
- For each method, the match threshold is the next representable number above
  the **largest absent-trial top score in validation**. Thus zero false accepts
  are observed on 12 validation absences by construction. This is a deliberately
  conservative calibration rule and does **not** establish a population false
  accept rate. The threshold is frozen before evaluating S04.

The local AICity22 dataset license permits academic, non-commercial research
and restricts redistribution. The extracted frames and raw dataset are not
committed. Share only the split definition, code, and aggregate findings unless
the dataset's terms separately allow an artifact.

## Measured result

| Method | Validation threshold | Test present Rank-1 | Correct test handoffs | Wrong-ID handoffs on present | Missed present | False accepts on absent | Correct absent rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pi 140-D fingerprint | 0.9855 | 7/12 | 0/12 | 2/12 | 10/12 | 2/12 | 10/12 |
| Pretrained FastViT T8 | 0.8032 | 6/12 | 3/12 | 0/12 | 9/12 | 0/12 | 12/12 |

The test has 12 present and 12 absent trials and a mean of 3.92 annotated
gallery vehicles. Validation has the same present/absent counts and a mean of
2.83 candidates. Rank-1 measures retrieval when the target is present, before
thresholding. A wrong-ID handoff accepts another vehicle; a false accept on
absent accepts any vehicle when the target is absent. The small sample makes
these counts unstable. In particular, zero false accepts in 12 test absences
does not prove that the model is safe at a low operating FAR.

Neither baseline is sufficient for reliable handoff at this conservative
operating point. The next substantive question is whether training a compact
cross-category identity encoder on **additional, identity-labeled data** can
improve held-out category matching and absent rejection, and whether that
encoder survives IMX500 conversion. This vehicle-only smoke test is a fixed
regression slice; do not select architecture or thresholds on S04 and then
report S04 as untouched test evidence.

## Reproduce locally

Set the three paths to your local checkouts and benchmark manifest. `PXL_ROOT`
points to `Pxl-loc-tak`; `SEG_ROOT` points to `seg_tracking`.

```bash
python -m pip install -r on_sensor_handoff/requirements-benchmark.txt
export PXL_ROOT=/home/megrad/Documents/GitHub/Pxl-loc-tak
export SEG_ROOT=/home/megrad/Documents/GitHub/seg_tracking
export BENCHMARK_MANIFEST="$SEG_ROOT/aicity_handoff_benchmark/outputs/manifest.jsonl"

python on_sensor_handoff/tools/import_aicity.py \
  --source-manifest "$BENCHMARK_MANIFEST" \
  --split on_sensor_handoff/splits/aicity_smoke_v1.json
python on_sensor_handoff/tools/validate_manifest.py \
  on_sensor_handoff/data/aicity_smoke_v1/observations.jsonl --check-files
python on_sensor_handoff/tools/run_aicity_baselines.py \
  --model "$SEG_ROOT/models/onnx/timm/reid_fastvit_t8.apple_in1k_256.onnx" \
  --fingerprint-source "$PXL_ROOT/coms_servers/opencv_contour_tracker/src/camera/lightweight_fingerprint.cpp" \
  --fingerprint-include "$PXL_ROOT/coms_servers/opencv_contour_tracker/include"
```

The runner needs `g++` and writes detailed scores and aggregate metrics to
ignored `on_sensor_handoff/runs/aicity_smoke_v1.json`. If the benchmark
manifest hash changes, the importer stops so the frozen split can be audited.
