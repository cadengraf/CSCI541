# Aerial data you can run yourself

## Research priority: match the labels to the claim

Use one benchmark per claim. A crop identity label supports ReID; a track ID
within one video supports same-video retrieval; a pixel mask supports foreground
quality. None substitutes for another. Keep official test IDs out of training
and checkpoint selection. More frames from one track do not add independent IDs.

| Priority | Dataset | Best role | Access and protocol caution |
| --- | --- | --- | --- |
| 1 | [VRAI](https://github.com/ASGO-MM/VRAI-Dataset) | Aerial **vehicle** ReID paper comparison; 137,613 crops, 13,022 IDs | Official train/test identity partitions and test query/gallery order. The public annotation description lists test image names and order but no test ID mapping; use the official evaluator if labels are unavailable. Noncommercial use. Keep the complete official test partition out of training. |
| 1 | [VRU](https://github.com/GeoX-Lab/ReID/tree/main/VRU) | Additional aerial vehicle identity training; 172,137 images, 15,085 IDs | Five UAVs and varied weather/lighting. Verify its released split and camera metadata before claiming a cross-drone result. |
| 1 | [iSAID](https://captain-whu.github.io/iSAID/index.html) | Aerial **instance mask** training and held-out mask evaluation; 655,451 objects in 2,806 scenes | Fifteen classes, including vehicles and aircraft. Convert each instance to a binary object-crop mask before using the current foreground evaluator. It has no ReID identities. |
| 1 | [UAVid](https://uavid.nl/) | Oblique drone **semantic** mask transfer for people and cars | Eight scene classes; human, moving car, and static car are useful foreground categories. Same-class adjacent objects cannot be separated by these labels; it is not an instance-mask benchmark. |
| 2 | [VisDrone MOT](https://github.com/VisDrone/VisDrone-Dataset#task-4-multi-object-tracking) | Aerial track pretraining and same-video diagnostic; already local | IDs are sequence-specific; they do not establish that an object appears in another drone. Boxes are not segmentation masks. |
| 2 | [BuckTales](https://edmond.mpg.de/dataset.xhtml?persistentId=doi%3A10.17617%2F3.JCZ9WK) | Test-only two-drone **animal** ReID; already local | The same animal may recur across pairs/days without a verified global ID. Keep it out of training. Existing local scores are exploratory because the test set has been inspected. |
| 2 | [DIVOTrack](https://github.com/shengyuhao/DIVOTrack) | Cross-view **person** tracking/ReID, with a drone view in the older release | Check the precise release and identity mapping; the authors require access approval. |
| 3 | [AG-VPReID](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_AG-VPReID_A_Challenging_Large-Scale_Benchmark_for_Aerial-Ground_Video-based_Person_Re-Identification_CVPR_2025_paper.pdf) | High-altitude aerial-ground **person** transfer; 6,632 subjects, 32,321 tracklets | Video benchmark with 15–120 m drone heights. Implement its published tracklet protocol before comparing scores. |
| 3 | [UAV-VeID](https://github.com/tengshangzhi/UAV-VeID) | Second aerial vehicle ReID test | Smaller than VRAI/VRU and requires a research agreement. |

Practical order: obtain VRAI and VRU for vehicle identity learning/evaluation;
obtain iSAID masks for the foreground branch; retain BuckTales as the separate
animal stress test. If the thesis is about people, use an official aerial-person
protocol. A single "any object" result would need separate category evidence.

### Add VRAI without test leakage

The official VRAI `train_annotation.pkl` and extracted `images_train/` are in
`data/aerial/vrai/`. Run `python prepare.py vrai` to verify the training images
and write `data/aerial/vrai/train.csv`. It also writes an identity-disjoint
custom split: `train_disjoint.csv`, `calibration_query.csv`,
`calibration_gallery.csv`, `query.csv`, and `gallery.csv`. The split uses 5,042
training IDs, 630 calibration IDs, and 630 final test IDs, with camera 0001
queries and camera 0002 galleries. It verifies
that every filename is in the official `train_im_names` list and records the
camera in the filename. The official test partition is never included. Only
unpickle the archive from the trusted authors; Python pickle files can run code.
The VRAI disjoint training source is active in `python train.py`. Its config
starts from the pre-VRAI balanced checkpoint and writes
`joint_aerial_validated.pt`; the completed aerial-expanded weights remain
available. VRAI calibration mAP selects the checkpoint; the final held-out IDs
are used only for evaluation. This is **not** the published VRAI test split:
its public test metadata lacks identity labels for local Rank-1/mAP.

### Add VRU and use the UAVid toolkit

VRU now lives under `data/aerial/vru/source/`. `python prepare.py vru`
created the official identity-disjoint VRU training manifest. The 8,000
official test identities are excluded. VRU remains train-only in `config.yaml`;
VRAI now uses a disjoint custom split of its official training IDs.

The [UAVidToolKit](https://github.com/YeLyuUT/UAVidToolKit) source is downloaded
under `capstone/reid/utils/UAVidToolKit`. It provides label conversion,
visualization and semantic-segmentation evaluation scripts. It is **not** the
UAVid image dataset, which is not present in this repository. The toolkit's
fast Cython evaluator needs a separate build; its basic label tools are plain
Python. Do not run its evaluator against the joint model until the eight UAVid
classes have been mapped to this model's mask outputs.

The `data/aerial/isaid/` folder contains train/val annotations and mask ZIPs, but the
original matching train/val images were absent. `python prepare.py isaid` obtains
a small set of dimension-checked DOTA validation mirror JPEGs and builds a
16-crop iSAID validation mask diagnostic. It does not add iSAID to training.

### Measure the aerial foreground head

`python benchmark.py masks` reads `aerial_mask_eval` in `config.yaml` and evaluates
the saved checkpoints on `data/aerial/masks/eval.csv`. Each image is one iSAID
validation object crop. Its binary mask comes from an official iSAID instance
polygon, and the scene image is a dimension-matched, re-encoded DOTA mirror
JPEG. The CSV records image, mask, category, scene, and annotation ID. The
command prints foreground IoU and Dice overall and by category. This small
sample is a local diagnostic; it is not representative of people or animals,
and it is not an official iSAID benchmark. VOC mean IoU is a different metric.

```bash
cd capstone/reid
python prepare.py isaid
python benchmark.py masks
```

The current checkpoint was trained on catalog photos and VOC masks. VisDrone has been downloaded, and the project now has a local AICity cross-camera vehicle benchmark as an additional training source. AICity uses fixed traffic cameras, so it adds cross-view identity supervision but does not establish cross-drone accuracy. The other listed sources are options for future data collection. The new `python train.py` configuration combines VisDrone, AICity when prepared, SOP, ABO, and VOC while reporting the held-out retrieval results separately. The VisDrone preparation script creates CSV manifests without copying images. The AICity converter extracts compact crops from the local videos for efficient training.

## Start with identity learning

For object ReID, the key labels are **the same physical object in multiple images**. A pretrained ShuffleNetV2 backbone can turn each detected object crop into an embedding. Training with those identity labels does not require segmentation masks. A detector is still needed to find objects in full aerial frames; its boxes are enough for a first ReID experiment. Instance masks are optional when precise outlines matter. The current video uses VisDrone ground-truth boxes; there is no full-frame detector in this project. Training more ReID data alone will not supply detections.

Set `reid_only: true` in [config.yaml](config.yaml), then run `python train.py` to train without VOC mask batches. Set it back to `false` for joint training. Run `python benchmark.py` to evaluate saved checkpoints on the configured retrieval sets. Global and mask-guided pooling are separate comparison rows, so use the same checkpoint and identity split when comparing them.

## Which labels are useful?

| Dataset | What its labels mean | Use in this code | Access |
| --- | --- | --- | --- |
| [VisDrone MOT](https://github.com/VisDrone/VisDrone-Dataset#task-4-multi-object-tracking) | The same person or vehicle tracked across drone video frames | **Object ReID** via tracked bounding-box crops | Official train and validation archives; about 7.53 GB and 1.48 GB compressed |
| [UAVDT](https://sites.google.com/view/daweidu/projects/uavdt) | Vehicle tracks in UAV video | **Vehicle ReID** from repeated tracked boxes, if the released MOT labels are used | Author download links; verify ID conventions and splits before making manifests |
| [VRU](https://github.com/GeoX-Lab/ReID/tree/main/VRU) | The same vehicle across UAV views | **Object ReID** via the generic CSV format below | Author provides Baidu and Google Drive links; about 172,000 labeled vehicle images |
| [UAV-VeID](https://github.com/tengshangzhi/UAV-VeID) | The same vehicle in aerial images | **Object ReID** via the generic CSV format below | Provider requires a signed research agreement |
| [BuckTales](https://edmond.mpg.de/dataset.xhtml?persistentId=doi%3A10.17617%2F3.JCZ9WK) | The same blackbuck antelope across two drones | **Animal ReID** from the provider's cross-drone ID mappings | Public dataset; ReID archive is about 60 GB |
| [DIVOTrack](https://github.com/shengyuhao/DIVOTrack) | The same pedestrian across overlapping moving camera views; an older release includes a drone view | **Cross-view person ReID** via its supplied ReID format | Authors require a license request for archive passwords |
| [University-1652](https://github.com/layumi/University1652-Baseline) | The same *building/location* in drone and satellite views | **Cross-view location retrieval**, not individual-object ReID | Request access from the authors |
| [DenseUAV](https://github.com/Dmmm1997/DenseUAV) | The same *location* in drone and satellite views | **Cross-view location retrieval** through its identity folders | Download linked from the author repository |
| [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) | The same *scene/location* at drone heights and in satellite imagery | **Cross-view location retrieval** | Author download links; academic research only |

More aerial labels exist for the other parts of the pipeline:

| Dataset | Annotation | What it could improve |
| --- | --- | --- |
| [iSAID](https://captain-whu.github.io/iSAID/index.html) | Instance masks for 15 aerial object classes | Aerial instance detection and masks |
| [UAVid](https://uavid.nl/) | Semantic labels for eight urban drone-scene classes | Aerial semantic segmentation |
| [LoveDA](https://github.com/junjue-wang/loveda) | Seven land-cover labels in satellite/aerial images | Land-cover segmentation; academic use restrictions apply |
| [DOTA](https://captain-whu.github.io/DOTA/dataset) | Oriented aerial object boxes | Aerial object detection; academic use restrictions apply |
| [SpaceNet](https://spacenet.ai/datasets/) | Building footprints and road labels | Satellite building/road masks; check each release's license |

[FairMOT](https://github.com/microsoft/FairMOT) and [CrossMOT](https://github.com/shengyuhao/DIVOTrack) are **model methods**, not additional datasets. FairMOT learns detection and identity features together; CrossMOT is the baseline model introduced with DIVOTrack. They are useful architecture references, but adding their names to a dataset list would not add training images. BuckTales and DIVOTrack do add identity-labeled images after you obtain and convert them.

For a zero-shot BuckTales test of the current checkpoint, use the prepared observations described in [README.md](README.md) and run `python demo.py crossdrone`. It compares mask-guided and global pooling and saves a selected-object MP4. BuckTales is under `data/bucktales/`; its identity labels are not verified across recording days, so keep it test-only.

The second table is a **source catalog**, not a claim that those datasets feed directly into `python train.py`. The mask head in `utils/core/model.py` currently has VOC's 21 labels, and it has no full-image detector. Aerial mask training requires a class mapping or a larger head, and aerial full-image detection requires detector training. Detection boxes, land-cover classes, or building footprints must not be assigned fake ReID identities.


## Existing cross-camera vehicle data: AICity

Your local `aicity_handoff_benchmark/outputs/manifest.jsonl` contains physical vehicle IDs, source and destination cameras, boxes, and raw video paths. Run `python prepare.py aicity` from `capstone/reid` to turn its training partition into identity crops. The validation partition is divided into disjoint calibration and test IDs using the manifest's `benchmark_split` field. Calibration selects the best epoch by cross-camera mAP; test is reported after training. The converter also samples more labeled frames for the training vehicles from each video's `gt/gt.txt`. This uses the existing benchmark and videos; it is not a drone dataset or an official benchmark protocol. `python train.py` automatically includes AICity when all five manifests exist.

The cross-camera evaluation excludes every gallery crop from the query's camera and reports Rank-1 and mAP. The model's current VOC semantic mask is still unrelated to the selected AICity vehicle boundary; actual aerial instance-mask training remains future work.

## First aerial object experiment: VisDrone MOT

The first command below downloads the official [VisDrone-MOT train and val archives](https://github.com/VisDrone/VisDrone-Dataset#task-4-multi-object-tracking) from the provider's Google Drive links if they are missing, then extracts them. These are large downloads: about 7.53 GB and 1.48 GB compressed. If Google Drive reports a quota error, the script tries a [third-party Hugging Face copy](https://huggingface.co/datasets/myn0908/visdrone-dataset/tree/main) of the same ZIP filenames. This copy is not published by the VisDrone team; its archive structure is checked during extraction, but no official checksum is available here to prove byte-for-byte equivalence. Set `allow_third_party_mirror: false` in `config.yaml` if you want only the provider's links. You can also download ZIPs in a browser and save them as `data/aerial/VisDrone2019-MOT-train.zip` and `data/aerial/VisDrone2019-MOT-val.zip`; the script will extract those without downloading again. The converter reads track IDs from the annotations, samples frames at a configurable interval, and crops boxes **when loaded** rather than writing thousands of duplicate images. A track ID is namespaced by sequence.

The simple workflow reads every setting from [config.yaml](config.yaml). After `pip install -r requirements.txt`, run this from `capstone/reid`:

```bash
python prepare.py visdrone         # prepare manifests, then measure the baseline; no training
python train.py visdrone   # train a separate ReID checkpoint, then evaluate it
```

Run the first command and check its counts and Recall@1 before starting the second. The second uses the `visdrone` settings in `config.yaml` and leaves `joint_all_current.pt` untouched. It trains on VisDrone identity crops only; SOP, ABO, and VOC are not mixed into that aerial checkpoint. The longer commands below are equivalent manual controls for the first experiment.

The short workflow evaluates both checkpoints with **global average pooling**, including the baseline loaded from the older joint checkpoint. This keeps its before/after retrieval comparison on the same embedding path. The original joint checkpoint normally uses its VOC mask to weight features in the PC pipeline, so its mask-guided score may differ from this aerial baseline.

From `capstone/reid`, after installing `requirements.txt`:

```bash
python prepare.py visdrone-folders \
  --train-root /path/to/VisDrone2019-MOT-train \
  --val-root /path/to/VisDrone2019-MOT-val \
  --output-dir data/aerial/visdrone
```

The default uses every tenth frame, boxes at least 16 pixels wide and high, and up to 12 training observations per track. Validation uses early and late observations of the **same** track, at least ten frames apart. This measures same-video tracking appearance, which is easier than recognizing an object across unrelated cameras. The manifest uses the official annotation columns `frame,target,x,y,width,height,score,category,...`; ignored boxes and invalid IDs are skipped.

For the one-source VisDrone experiment, `python prepare.py visdrone` runs preparation and baseline evaluation. `python train.py visdrone` trains its configured checkpoint and evaluates it. For a mixed-source run, add the prepared VisDrone manifests to `sources` in [config.yaml](config.yaml), then run `python train.py`. VisDrone tracking boxes are **not** segmentation masks. Keep the original checkpoint and compare results on every relevant source before using new weights. If the checkpoint changes, rebuild any appearance gallery with that checkpoint.

## Drone-to-satellite location experiments

After obtaining [University-1652](https://github.com/layumi/University1652-Baseline), use its documented `train/drone/<building>`, `train/satellite/<building>`, `test/query_drone/<building>`, and `test/gallery_satellite/<building>` folders:

```bash
python prepare.py folders \
  --root /path/to/University-1652 \
  --output-dir data/aerial/university1652
```

The same command can read [DenseUAV](https://github.com/Dmmm1997/DenseUAV) if its extracted folder names match. Use `--train-drone`, `--train-satellite`, `--query`, and `--gallery` to supply different subfolder names. Results are **location retrieval** scores; they do not tell you whether a car or person was re-identified.

After obtaining the raw [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) dataset (`drone_view_512/<scene>/<height>/...` and `satellite-view/<scene>/...`):

```bash
python prepare.py sues \
  --root /path/to/SUES-200-512x512 \
  --output-dir data/aerial/sues200 \
  --test-scenes 40 --seed 42
```

This creates a **custom scene-level holdout**. It is not the published SUES-200 benchmark split, so do not compare its score with published benchmark numbers. Add `--heights 150 200 250 300` to select heights explicitly; the default includes all available heights.

To train either location dataset, add its train, query, and gallery CSVs as a source in `config.yaml`, then run `python train.py`. Keep the held-out query and gallery manifests out of training.

## Other identity datasets

The `sources` section of `config.yaml` accepts your own CSV manifests for identity-labeled data such as VRU, UAV-VeID, or DIVOTrack. `utils/core/datasets.py` reads each file with `path,id` columns; optional `x,y,w,h` columns crop an object from a full frame. Paths may be absolute or relative to the CSV. Supply three files for **train**, **query**, and **gallery**. Every training ID needs at least two images; query and gallery need matching IDs and must not contain the same image crop. Use the source provider's official train/test split, and keep track or object IDs distinct across unrelated videos. BuckTales remains test-only in this project. Do not convert category labels such as “car” into identity labels: thousands of different cars share that category.

The same manifest format can represent another aerial vehicle ReID dataset once you obtain its images and identity labels. Run `python prepare.py --help` to see the preparation commands and `python train.py --help` to see training modes.

When you check back in, the useful things to share are the preparation counts (photos and IDs), the **before-training** aerial Recall@1, the score after one epoch, and a few actual full-image predictions. Those show whether to add another identity dataset, improve small-object detection, or work on aerial masks next.
