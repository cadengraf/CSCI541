# Fixed-model tracking experiments

The teacher weights stay fixed. These experiments change temporal association
and mask propagation; they do not train or improve the cross-camera embedding.

## Commands

Run from `capstone/reid` with your normal PC `.venv` active. Dependencies, source,
and released EdgeTAM/CUTIE weights are installed on this machine.

```bash
python benchmark.py trackers
```

This command first launches the teacher's isolated environment to cache raw
predictions, then runs all four tracker variants in the PC environment. It scores
all 2,981 frames in the nine conventional KITTI MOTS validation sequences using
the existing official TrackEval adapter. No ground truth is supplied to a tracker.
The final comparison will be `metrics/tracking/summary.html`; timings are in
`metrics/tracking/timing.csv`. This is a full GPU evaluation, not training.

For SSH:

```bash
tmux new -s vehicle-tracking
python benchmark.py trackers
```

Detach with **Ctrl+B**, then **D**. Reattach with
`tmux attach -t vehicle-tracking`. Repeating the benchmark resumes completed
cache frames and completed tracker sequences. A partially tracked sequence
restarts from its first frame so temporal state remains correct. Do not run two
writers to these output paths at once.

Short integration check, without publishing benchmark scores:

```bash
python benchmark.py trackers-check
```

This uses 40 frames from each of two sequences in the same process, injects one
empty-detection frame per sequence, exercises repeated propagator initialization,
mask refresh and lost-track aging, then evaluates temporary RLE files. It must not be interpreted
as a full dataset score. On another machine, first run:

```bash
python benchmark.py trackers-setup
```

The teacher environment/data must already be prepared. Setup fetches pinned
third-party repositories and validates SHA-256 hashes of released propagation
weights. EdgeTAM/CUTIE also initialize their published backbone models; the first
model construction can download those backbone weights to PyTorch/HF caches.

## What is being compared

Edit [`configs/tracking.yaml`](../configs/tracking.yaml).

| Variant | Association | Appearance | Propagated masks |
| --- | --- | --- | --- |
| `baseline` | Existing velocity/box-overlap Hungarian tracker | Teacher vehicle embeddings | None |
| `baseline_motion` | Same tracker, appearance weight zero | Disabled | None |
| `mcbyte_no_masks` | Upstream McByte++ Kalman filter, staged matching, camera-motion compensation and export-ID reconnection | Same cached teacher vehicle embeddings | Disabled |
| `mcbyte_edgetam` | Same McByte++ adapter | Same cached teacher vehicle embeddings | Released EdgeTAM cues |

The McByte++ **person** ReID network is replaced with an adapter that retrieves
the teacher embedding belonging to the actual matched detection. It never
recomputes an embedding with a different ReID network. A high-IoU guard prevents
borrowing another candidate's embedding. This is an adapted vehicle system,
not an exact reproduction of the paper's sports model/results.

Camera motion uses the upstream sparse optical flow implementation. Its pinned
version initializes affine-validity guards only for ORB despite using them for
sparse flow too; the adapter supplies the same guard values to that path. The
upstream source files are left intact. Its mask manager also has an empty-start
return mismatch, handled in the adapter. Original KITTI PNGs are read lazily,
with video and temporal state offloaded to CPU, avoiding duplicate JPEG folders
or loading the entire sequence into GPU memory. CUDA connected-component
postprocessing is disabled; the core EdgeTAM model and mask propagation remain.

All methods share the same teacher checkpoint, geometry, embeddings, raw scores
and binary predicted masks. The cache retains candidates down to 0.1 confidence.
The baseline keeps its original 0.3 threshold; McByte++ can use low-confidence
candidates in its second association stage and has its upstream new-track
threshold of `track_thresh + 0.1`. Differences in output detections are therefore
part of the tracking policy. This is not a comparison with identical final
retained boxes.

**Every exported segmentation mask comes from the teacher.** EdgeTAM masks guide
association but do not replace the exported instance masks in this experiment.
This isolates temporal mask cues from substituting a different segmenter.
Overlapping output pixels are assigned by confidence, as in the original
evaluation. McByte++ young export IDs are finalized after processing the
sequence, following its demo's buffering approach; report that delay rather than
claiming immediate, immutable online IDs.

The report gives HOTA, DetA, AssA, IDF1, sMOTSA, MOTSA, MOTSP, identity switches,
TP, FP and FN. Check whether adding appearance helps relative to motion only,
and whether temporal masks help relative to McByte++ without masks. All tracking
thresholds are initial settings, not values optimized on the validation labels.
Any later tuning must be reported and final claims need an untouched test set.

### Timing

`tracking_replay_ms` includes image/cache loading, mask decoding, propagation,
and association, with CUDA synchronized. `cached_teacher_ms` measures the prior
image-load/network pass. `estimated_total_ms` adds these costs and consequently
includes some duplicate loading; it is **not measured live end-to-end latency**.
Model startup and final text export are excluded. Compare hardware fields and
timing scopes, not just the smallest number. Measure a live combined deployment
pipeline before claiming an FPS gain or a deployment advantage.

The compact internal `.pt` records live in `data/tracking/teacher_cache/` and
contain compressed RLE masks, boxes, scores, embeddings and inference timings.
They are evaluator inputs, not visual outputs. No PNG, JSON or NPZ dumps are
created. A checkpoint-hash/config guard prevents silently reusing another model's
cache. Completed-sequence guards also prevent mixing tracker settings.

## Selected-object CUTIE video

```bash
python demo.py cutie
```

Output: `data/videos/selected_cutie.mp4`. The current default is a 200-frame,
20-second clip from validation sequence `0010`, with the largest predicted
vehicle selected at the start. There are no ground-truth mask prompts.

- **Left:** teacher detections/masks associated by the current baseline tracker.
- **Right:** released CUTIE initialized using that exact predicted object mask,
  then propagating the selected mask through subsequent images.

CUTIE's network does not use the teacher ReID head in this video. It has no
automatic cross-camera matching or detector-based reacquisition. A visible green
mask is a prediction, not evidence that the correct object is being followed.
The preview is qualitative and does not establish a ranking between methods.

To choose a particular vehicle over SSH, edit `selected.click_xy: [x, y]` using
coordinates in the original image. Set `selected.selection: click` for a local
desktop click interface instead. Other settings control sequence, start frame,
frame count, output FPS and internal CUTIE resolution. Changing these does not
change the teacher weights. CUTIE is an additional model and its compute must
be counted in any eventual deployment comparison.

## Code map

All implementation lives under `utils/tracking/`:

| Module | Responsibility |
| --- | --- |
| `setup.py`, `requirements.txt` | Source, dependencies and released weights |
| `common.py` | Config, source revision checks and compressed prediction records |
| `cache.py` | Fixed teacher export in its isolated environment |
| `mcbyte.py` | Vehicle embedding adapter and upstream tracker integration |
| `propagation.py` | Lazy EdgeTAM frame loading and CUTIE model loading |
| `comparison.py` | Sequence replay, RLE export, timing and official evaluation |
| `video.py` | Selected-object side-by-side MP4 |

Pinned external code stays under ignored `utils/vendor/` folders. Root commands
remain `benchmark.py` and `demo.py`.

Sources: [McByte++](https://github.com/tstanczyk95/McBytePlusPlus),
[EdgeTAM](https://github.com/facebookresearch/EdgeTAM),
[CUTIE](https://github.com/hkchengrex/Cutie).
