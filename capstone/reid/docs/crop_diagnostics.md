# Crop preprocessing diagnostic

Run from `capstone/reid` with the PC virtual environment active:

```bash
python benchmark.py crop-diagnostics
```

Configure `configs/crop_diagnostics.yaml`. This command launches the isolated
teacher environment and performs inference only. No production settings or
weights change.

Four variants separate geometry from resolution:

| Variant | Geometry | Maximum side / square size |
| --- | --- | --- |
| aspect_384 | Current aspect-preserving resize | 384 |
| square_384 | Stretch width and height to a square | 384 × 384 |
| aspect_256 | Aspect-preserving resize | 256 |
| square_256 | Stretch width and height to a square | 256 × 256 |

At each size, aspect versus square changes geometry; comparing sizes also
changes spatial resolution. Square images are bilinearly resized from the
original decoded RGB tensor, then passed through the unchanged normalization
and padding path. This tests geometry/size, not an exact reproduction of
TransReID's complete preprocessing, interpolation, normalization or architecture.

All variants use the same completed distilled joint checkpoint, 64 seeded VRAI
calibration identities and all their query/gallery crops (default seed 42 matches
the earlier pooling diagnostic). The existing data loader checks train,
calibration and final identity separation. The smaller gallery makes these
scores easier than full calibration or final scores. Neither final test nor
training images are embedded by this diagnostic.

Both predicted-mask and global pooling are measured using the same shared
features for each variant. Timing includes disk decode, CPU/GPU transfer,
feature extraction, both pooling modes and progress printing; initialization is
outside the timer but first-use kernels are not excluded. Treat this as coarse
extraction cost, not benchmark-quality model latency. Padding and aspect ratio
change the amount of computation.

Outputs:

- `metrics/crop_diagnostics/summary.html` — comparison table and caveats
- `metrics/crop_diagnostics/summary.csv` — metrics and extraction cost
- `metrics/crop_diagnostics/provenance.yaml` — checkpoint hash, calibration IDs,
  manifest hashes, settings, hardware and torch version

No images, NPZs, feature caches or trained checkpoints are written. A failed run
can leave a partial summary CSV; rerun the command to regenerate all variants.
The HTML/provenance are written only after all variants finish. Repeated runs
replace these diagnostic outputs.

A benefit from square inference suggests a matched training-preprocessing test.
No benefit rules out a simple inference resize fix; it does not establish that
training with another preprocessing policy cannot help. Differences on this
small calibration sample are development evidence, not a new blind test.

## Full calibration validation

```bash
python benchmark.py crop-validation
```

`configs/crop_validation.yaml` compares the original `aspect_384` policy against
the selected `square_256` policy on **all 630 calibration identities**. Null
`calibration_identities` means all available calibration IDs. Outputs go to
`metrics/crop_validation/`, preserving the exploratory subset results. This
comparison changes geometry and size together; the earlier factorial diagnostic
separates those factors. It is still development validation: calibration has
informed checkpoint and preprocessing selection, so it is not an independent
final test. No training settings, model weights or final-test embeddings change.
