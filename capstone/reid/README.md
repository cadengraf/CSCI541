# ShuffleNetV2 object masks and ReID

**Start here:** [ARCHITECTURE.md](ARCHITECTURE.md) shows the model diagram,
training path, module map, command outputs, and where to make changes. The
the four root commands are [train.py](train.py), [prepare.py](prepare.py),
[demo.py](demo.py), and [benchmark.py](benchmark.py). Their implementation
is grouped by purpose under `utils/`. Edit [config.yaml](config.yaml), then
run the command for the work you want to do.

## Data layout

All dataset files live under `data/`: VRAI in `data/aerial/vrai/`, VRU in
`data/aerial/vru/source/`, iSAID in `data/aerial/isaid/`, and BuckTales in
`data/bucktales/`. Existing VisDrone, VOC, SOP, ABO, AICity, evaluation crops,
videos, and reports also remain under `data/`. The extracted datasets were
checked before their redundant download archives were removed. VRAI's
`train_annotation.pkl` remains because `python prepare.py vrai` uses it to rebuild the
identity manifests; iSAID's mask ZIPs remain as the original annotation data.
Model weights and training recovery states live in `checkpoints/`.

For the research comparison, see [aerial dataset priorities and preparation](AERIAL_DATA.md).
`python prepare.py vrai` now splits the official VRAI **training** identities into
disjoint train, calibration, and final cross-camera test IDs (5,042/630/630).
It preserves the original `train.csv` used in the completed run.
`python prepare.py isaid` creates a small local foreground diagnostic from
official iSAID validation polygons and dimension-checked DOTA mirror images;
`python benchmark.py masks` compares saved mask heads on those crops. Neither result is an
official VRAI or iSAID benchmark score. The next method comparison still
requires OSNet, TransReID, and a separately trained mask pipeline on an
identity-disjoint protocol with measured latency on the same hardware.

Research comparison protocol: train all ReID methods on the same official
training IDs, hold out the same query/gallery identities, and report Rank-1,
mAP, parameter count, and median per-crop latency on one named machine and
input size. Include a ReID-only ShuffleNetV2 arm, the joint model with global
pooling, the joint model with mask pooling, [OSNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html),
and [TransReID](https://openaccess.thecvf.com/content/ICCV2021/papers/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.pdf).
For a two-model cost comparison, run a separately trained foreground model
followed by the same ReID backbone and count both models' time and parameters.
Train the foreground head on target-domain instance masks and measure its
held-out IoU. The existing frozen ImageNet rows do not answer this question.

This project trains **one ShuffleNetV2 x1.5 model for an object crop**:

```text
object crop → shared ShuffleNetV2 backbone ┬→ 21-class semantic mask
                                          └→ ReID feature map
mask foreground + feature map → mask-guided pooling → 128D embedding
128D embeddings → appearance matching
```

`python train.py` uses [config.yaml](config.yaml) to train the shared backbone, mask head, and ReID head. The ReID loss combines **identity classification and batch-hard triplet loss**. Batches contain 16 physical IDs with four images each; when camera labels are available, the sampler prefers images from different cameras. The training-only identity classifier is omitted from the saved inference model. VOC supplies semantic segmentation masks; SOP, ABO, VisDrone, AICity, VRAI, and VRU now supply identity crops. VOC's semantic foreground is not a selected-object instance mask. The completed aerial-expanded run started from `joint_balanced.pt`; VRAI/VRU official test identities remained excluded. The next run again starts from `joint_balanced.pt`, made before VRAI was added, so its new VRAI holdout identities have not appeared in the starting weights.

The aerial-expanded run completed 12 epochs, but preservation-based selection kept **epoch 2**. On the local AICity test, its Rank-1/mAP is **13.79%/4.02%**, versus **13.39%/3.88%** for the balanced checkpoint. On the exploratory BuckTales test, it is **31.21%/32.27%**, versus **33.53%/33.48%**. The earlier multiview checkpoint remains stronger on both retrieval tests (**27.47%/7.46%** AICity; **38.44%/39.19%** BuckTales), but its VOC mIoU is lower. The new model's VOC mIoU is **33.33%** versus **32.70%** for balanced. These are local crop-retrieval results, not official VRAI/VRU test scores or proof of aerial mask quality. The updated [benchmark report](data/benchmark.html) and PNG figures include the new checkpoint and its global-pooling ablation.

On the new 16-object iSAID validation diagnostic, the aerial-expanded,
balanced, and multiview checkpoints reach **35.55%, 38.38%, and 9.33%**
foreground IoU respectively. Large vehicles are especially weak (7.12% for
the expanded model). This tiny instance-mask sample uses four objects per
class and must not be read as a published segmentation benchmark. When two
objects share a crop, the current semantic foreground head can light up both;
it cannot reliably isolate the selected instance without instance-level
training or another selection mechanism.

## Prepare more data and train

From `capstone/reid`, with the project virtual environment active and `ffmpeg` installed:

```bash
python prepare.py aicity
python train.py
```

`python prepare.py aicity` reads your existing AICity handoff manifest and raw videos, samples labeled vehicle crops, and writes `train.csv`, `calibration_query.csv`, `calibration_gallery.csv`, `query.csv`, and `gallery.csv` in `data/aicity/`. Physical IDs in these three partitions are disjoint. Training uses the train partition; each epoch's cross-camera calibration mAP selects the best checkpoint; final cross-camera Rank-1 and mAP use the held-out test partition. This is a project-specific split of the local benchmark, not an official AICity benchmark score. The converter currently finds about 16,747 train crops from 184 IDs, 1,046 calibration crops from 145 IDs, and 4,591 test crops from 337 IDs. AICity is fixed-camera traffic footage, not aerial imagery. The same training run also includes the aerial VisDrone identities already in `data/aerial/visdrone/`. If you skip preparation, `python train.py` still runs with VisDrone, SOP, ABO, and VOC; it prints that AICity is unavailable.

The next run is configured for 12 epochs × 1,000 batches of 64, sampling identities across SOP, ABO, VisDrone, AICity, the VRAI train split, and VRU. VOC masks are trained every second update. It starts from `joint_balanced.pt` and writes `checkpoints/joint_aerial_validated.pt`. VRAI calibration mAP now has the largest checkpoint-selection weight. The final VRAI IDs are evaluated only after training. The stronger crop, lighting, blur, grayscale, and erasing augmentation is set in `config.yaml`.

If you train over SSH, use tmux so closing the terminal does not stop Python:

```bash
cd /home/megrad/Documents/GitHub/CSCI541/capstone/reid
tmux new -s reid
../.venv/bin/python -u train.py 2>&1 | tee -a data/training_validated.log
```

Press **Ctrl+B, then D** to detach. Reconnect with `tmux attach -t reid`. Training writes the best inference model to `checkpoints/joint_aerial_validated.pt` and full recovery state to `checkpoints/joint_aerial_validated.last.pt` after each completed epoch. If the process stops, rerun the command in tmux; it resumes from the last completed epoch and redoes only an interrupted epoch. Do not change training settings between restarts; the recovery file checks their signature.

Checkpoint selection now emphasizes VRAI cross-camera calibration mAP, alongside AICity mAP, VisDrone/SOP/ABO Recall@1, and VOC mean IoU. Its score is a weighted average of log improvement over the starting checkpoint. VOC must stay above 80% of its starting score. The checkpoint stores every epoch's selection metrics and a completion marker. VRAI train, calibration, and final test IDs are disjoint; the final query comes from camera 0001 and gallery from camera 0002. This is a **custom split of the official training partition**, not the official VRAI leaderboard protocol. After training, `evaluate_after_train` reports the held-out results. Expect a long run. The saved model keeps ShuffleNetV2 x1.5 and its 128D output.

The AICity preparation command reads its source files under `/home/megrad/Documents/GitHub/seg_tracking/aicity_handoff_benchmark` and `/home/megrad/Downloads/AICity22_Track1_MTMC_Tracking`; adjust `aicity.benchmark_manifest` if those files move. It writes crops under this project's ignored `data/aicity/` directory and does not change the original videos. The local AICity data and balanced 12-epoch run have been prepared and completed; the selected model is in `checkpoints/joint_balanced.pt`.

## Run the video

From `capstone/reid`, with the project virtual environment active and `ffmpeg` installed:

```bash
python demo.py
```

Open [data/reid_video.mp4](data/reid_video.mp4). The video reads VisDrone validation frames and **ground-truth boxes and IDs** to make object crops. Cyan shows the ShuffleNet model's predicted foreground mask. Each label shows a ground-truth ID and the closest same-class embedding from the previous sampled frame, with cosine similarity. Green means that closest ID is correct, orange means it is different, and blue means no prior same-class object was available. The model does not supply full-frame detections or the ground-truth IDs. The clip is a same-video diagnostic, not a cross-drone score.

Change the `video` section of [config.yaml](config.yaml) to select another sequence, checkpoint, or output path. `video.checkpoint: auto` uses the new joint checkpoint after training. The separate VisDrone ReID-only checkpoint uses global pooling and is not suitable for this mask demo.

## Select one object and search for it later

```bash
python demo.py select
```

The default `select` settings reproduce the saved example with track ID 14 and work over SSH. To choose another object by clicking, set `select.target_id: null`; a window then opens on the first annotated VisDrone frame. The command saves [data/selected_reid.mp4](data/selected_reid.mp4), replaying the same sequence from that frame. The model encodes your selected crop once, then compares it with every annotated candidate crop in each sampled frame. A temporal selector combines ReID similarity with predicted motion, box overlap, and size consistency to keep the match on one object. If no candidate passes its gate, it shows `TRACK LOST` rather than highlighting a distant lookalike. The cyan mask belongs to the tracked box. The video also shows the true target when the track is wrong and a running tracked-correct count. **VisDrone supplies candidate boxes and evaluation IDs; the model supplies masks and embeddings.** The selector uses no evaluation ID after the first-frame selection.

Use the `select` section in [config.yaml](config.yaml) to choose another validation sequence or set `target_id` to select an object by annotation ID without the click window. The click window requires a desktop display; the ID setting also works in a terminal without one. A new camera angle within the same sequence is a useful first check, but this does not measure cross-drone ReID; VisDrone track IDs are local to each sequence.

## Cross-drone evaluation and selected-object video

The [BuckTales ReIdentification dataset](https://edmond.mpg.de/dataset.xhtml?persistentId=doi%3A10.17617%2F3.JCZ9WK) has paired UAV videos and an **explicit mapping between local track IDs**. Its extracted files are under `data/bucktales/`. From `capstone/reid`, run:

```bash
python prepare.py bucktales inspect
python prepare.py bucktales prepare
python prepare.py bucktales audit
python demo.py crossdrone
```

`inspect` lists the 11 published drone pairs. `prepare` reads the official pair mappings and per-video tracks, samples four frames per matched identity and camera, and saves object JPEG crops plus `data/crossdrone/observations.csv`. It skips ambiguous ID links. The default `crossdrone.pair: null` prepares all 11 pairs. The command saves the manifest after each pair and skips completed pairs if restarted; a failed video leaves previously prepared pairs intact. It also removes old manifest rows that fall outside their annotated ID time ranges. Preparing a pair decodes its full videos and may take several minutes. Keep the original dataset folder in place to prepare other pairs later.

### BuckTales evaluation policy

**BuckTales is test-only in this project.** Its 11 pairs are not 11 independent recording groups: adjacent pairs in Sets 2–6 reuse the same middle-drone video. The provided IDs link animals within a paired recording, but do not provide verified identities across recording days. The [dataset authors](https://proceedings.nips.cc/paper_files/paper/2024/file/95286b5d4cd5b7953bd2bbe717300fe0-Paper-Datasets_and_Benchmarks_Track.pdf) say even field experts cannot identify these individuals between days. A pair-level 7/2/2 train/validation/test split would therefore leak recordings, while a day-level split would not certify identity separation. Run `python prepare.py bucktales audit` to see the shared videos and recording groups in your local copy.

No BuckTales training or validation split is generated. `train.py` rejects a source named BuckTales or a source manifest inside `data/crossdrone/`; train and select checkpoints using the other datasets. All BuckTales observations remain available for cross-drone evaluation through `python benchmark.py`. Because this test set has already been inspected during model development, report future scores as **exploratory test-only results**, not as a previously unseen final test. A publishable final estimate would need newly held-out recordings with verified physical IDs.

The CSV has one row per visible animal per sampled frame:

```csv
pair,camera,id,frame,path,x,y,w,h
set1_pair1,drone_A,animal_17,120,frames/drone_A_0120.jpg,300,200,48,60
set1_pair1,drone_B,animal_17,122,frames/drone_B_0122.jpg,615,384,50,64
```

`id` is the mapped physical animal ID **within that pair**; `camera` identifies one of its two drones. The prepare command writes `path` as an absolute object-crop JPEG and leaves `x,y,w,h` empty. If creating your own CSV, `path` can point to a full frame instead; `x,y,w,h` then crop the animal. Include other labeled animals in each destination frame as distractors. BuckTales does not provide an official ReID train/test split, so the command's numbers are a project-specific, zero-shot evaluation of the existing checkpoint.

With the CSV ready, run:

```bash
python demo.py crossdrone
```

The final command compares **mask-guided pooling and global pooling using the same trained weights**, reporting cross-camera Rank-1 and mAP per pair and overall. It excludes all gallery images from the query's drone. It also writes `data/crossdrone_reid.mp4`: a selected source-drone crop beside the top three matches from each sampled frame of the other drone. Change `crossdrone.pair`, `source_camera`, and `target_id` in `config.yaml` to choose the example. A correct match is green; wrong matches are red; cyan shows the model's predicted foreground. The video uses dataset boxes and IDs as candidate proposals and ground truth; the model supplies masks, embeddings, and ranking. Only the MP4 is written for visualization.

## Model comparison graphs

```bash
python benchmark.py
```

This writes [data/benchmark.html](data/benchmark.html) and standalone PNG figures in `data/benchmark_figures/`. The report has a **Rank-1-sorted local leaderboard for each test set**, with each row labeled as a matched experiment, earlier joint checkpoint, same-weights pooling ablation, or frozen ImageNet reference. Figures include leaderboard bars, Rank-1/mAP bars, CMC curves, per-pair BuckTales heatmaps, CPU speed versus accuracy, model size, VOC mask IoU, and pooling ablations for **both balanced and multiview checkpoints**. It evaluates the same AICity test crops and **every prepared BuckTales pair** for every model. BuckTales shows a pooled score and per-pair scores; each query searches only its own pair's other drone. It checks that AICity test IDs do not overlap training IDs. The report compares saved joint checkpoints and frozen ImageNet ResNet-18, MobileNetV3-Small, and EfficientNet-B0 feature extractors. The ImageNet baselines have **not** been trained for ReID; their scores show how generic features behave on these test sets, not a training-matched model competition. The BuckTales paper gives no standalone ReID baseline, and the official AI City Track 1 leaderboard measures full tracking with IDF1. Published scores from other datasets or tasks cannot be ranked beside these crop retrieval results.

The expanded reference set also includes official TorchVision weights for ShuffleNetV2 x1.0/x1.5, MobileNetV2, MobileNetV3-Large, EfficientNet-B2, RegNetY-400MF, ConvNeXt-Tiny, and Swin-T. The first benchmark run downloads missing weights; subsequent runs use the cache. `aicity_accuracy_latency.png` and `bucktales_accuracy_latency.png` plot ReID Rank-1 against **measured PC CPU latency in milliseconds on a linear horizontal axis** for all these model families and the project checkpoints. Corresponding `_zoom.png` files show the 0–25 ms range. Lines join size variants of the same frozen family. Every model receives the same 224×224 normalized crop and cosine retrieval protocol. These plots are analogous to architecture tradeoff figures but do not measure ImageNet classification accuracy, mobile latency, or Raspberry Pi AI Camera speed. Frozen ImageNet features are reference points; only the matched ReID-only and joint arms share a ReID training budget.

### Research comparison to build toward

The closest architecture reference is [Xu et al., *Joint Person Segmentation and Identification in Synchronized First- and Third-person Videos* (ECCV 2018)](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Mingze_Xu_Joint_Person_Segmentation_ECCV_2018_paper.pdf). It jointly learns segmentation and cross-view identity embeddings and uses soft foreground confidence to weight features. This project changes the backbone to ShuffleNetV2, uses aerial/cross-camera object crops, and predicts VOC semantic foreground instead of their person instance mask. It does not use their optical-flow stream. Their Table 1/2 compare mask IoU and identification mAP/accuracy with and without soft attention; Fig. 5 examines matching/segmentation across video conditions; Fig. 6 shows masks and cross-view identities. The corresponding project figures should report (1) matched ReID-only versus joint Rank-1/mAP, (2) mask-guided versus global pooling with identical weights, (3) **target-domain instance-mask IoU** alongside VOC IoU once ground-truth masks are available, (4) query/gallery success and failure examples, and (5) accuracy versus latency and parameters. Present these as analogous experiments, not as directly comparable numerical scores across their datasets and ours.

A published-method comparison needs your checkpoint evaluated on the **same official query/gallery split and metric** as those papers. [VRAI](https://github.com/ASGO-MM/VRAI-Dataset) is an aerial vehicle ReID benchmark; its [ICCV paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Vehicle_Re-Identification_in_Aerial_Imagery_Dataset_and_Approach_ICCV_2019_paper.html) reports multiple methods. The extracted training images and required annotation are under `data/aerial/vrai/`; unused dev/test archives were removed. The public test metadata does not expose physical IDs for a straightforward local Rank-1/mAP calculation. The local AICity and BuckTales scores are not VRAI leaderboard scores.

For a controlled ReID-only versus joint comparison, run:

```bash
python train.py compare
python benchmark.py
```

`python train.py compare` uses the common `checkpoints/joint_all_current.pt` start and the same ShuffleNetV2 backbone, identity datasets, batch size, seed, augmentation setting, optimizer settings, epochs, and number of ReID updates in both arms. The ReID-only arm uses global pooling and no mask head at inference. The joint arm uses mask-guided pooling and adds a VOC mask batch every two ReID batches, so its **total compute and image exposures are higher** even though the ReID training budget matches. The result measures the combined effect of mask training and mask-guided pooling. The separate pooling figures isolate the pooling choice with identical saved weights. Both arms use their final epoch, with no BuckTales-based checkpoint selection. Each completed epoch saves a full recovery state; rerun `python train.py compare` after an interrupted SSH session. The benchmark includes the matched arms only when both runs finish.

## AI Camera target

The current model runs on a PC. It has **not** been converted to an IMX500 `.rpk`. Conversion requires quantization and validation with the [Raspberry Pi AI Camera deployment tools](https://www.raspberrypi.com/documentation/accessories/ai-camera.html#model-deployment). Full-frame object proposals remain separate from the crop model. AICity measures cross-camera vehicle retrieval, while VisDrone measures same-video aerial retrieval. BuckTales is a zero-shot cross-drone evaluation; these checkpoints have not trained on its identities. The earlier one-pair BuckTales scores used some crops outside the published ID time ranges and should be replaced by a fresh benchmark after preparation.
