# Co-DINO, RF-DETR, and YOLO detector/instance-mask frontends

Mask R-CNN is removed from the active vehicle pipeline. The normal command uses Co-DINO ViT-L + RefineMask and automatically launches the existing compatible environment:

```bash
../.venv/bin/python demo.py vehicle
```

Device selection stays automatic. Live Co-DINO needs CUDA; offline cache replay remains available. Historical models, results, and training tools are retained as records, not exposed as a fallback in this pipeline.

## Released checkpoint accuracy

These are released-model COCO AP50:95 figures checked on November 16, 2026. They are not new vehicle measurements and not scores for our KITTI-fine-tuned checkpoint.

| Representative released models | Detection box AP | Instance-mask AP |
|---|---:|---:|
| Co-DETR / Co-DINO ViT-L instance model | 65.3 | 56.2 |
| RF-DETR-L / RF-DETR-Seg-L | 56.5 | 47.1 |
| RF-DETR-2XL / RF-DETR-Seg-2XL | 60.1 | 49.9 |
| YOLO26x / YOLO26x-seg | 57.5 | 47.0 |

Co-DETR figures come from its authors' [COCO instance model card](https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance), without test-time augmentation. RF-DETR figures come from the authors' [benchmark tables](https://github.com/roboflow/rf-detr). YOLO26 figures come from [detection documentation](https://docs.ultralytics.com/models/yolo26/) and [instance-segmentation documentation](https://docs.ultralytics.com/tasks/segment/). YOLO's detection figure is its ordinary detection result; the segmentation figure is the documented end-to-end mask result.

For RF-DETR and YOLO, the box and mask columns refer to separate detection and segmentation checkpoints. They are not the box AP and mask AP of one shared segmentation checkpoint. Training recipes, input resolutions, model capacity, and evaluation/postprocessing conditions differ, so the table is context rather than a controlled family ranking. Roboflow explicitly notes that its in-house measurements can differ from other vendors' reported results.

The original Co-DINO release configuration evaluates at scale `(2048, 1280)`. Our live/cache configuration uses short side 384 and maximum side 1280, plus KITTI/VRAI adaptation. Therefore the published 65.3/56.2 pair must not be presented alongside our measured runtime as one accuracy/speed operating point. RF-DETR-2XL detection uses 880-square input; Seg-2XL uses 768-square; the listed YOLO26 variants use 640-square. Small-vehicle performance is particularly sensitive to this difference.

## Practical interpretation for this system

Co-DINO/RefineMask is the established accuracy reference with working local weights, vehicle adaptation, and full-sequence tracking results. Its high reported COCO box and mask AP does not establish superiority on our vehicle videos.

RF-DETR offers real-time detection and dedicated instance-segmentation variants. RF-DETR-Seg-L and Seg-2XL are useful next candidates for the accuracy/latency comparison. YOLO26x-seg is a useful deployment-oriented instance-mask baseline. Both segmentation families already produce instance masks and boxes; adding a separate promptable segmenter is unnecessary for this comparison.

Keep the same TransReID specialist and local/global association when comparing frontends. The detector does not provide physical identity: it affects missed objects, localization, crop contamination, mask-based quality gates, and runtime. Measure:

- Vehicle box AP and recall, especially small or partially occluded cars.
- Vehicle mask AP and mask stability between frames.
- Tracking HOTA/DetA/AssA, IDF1, return recovery, and false reconnections.
- Full online cost on the same hardware, including decoding, preprocessing, detection/masks, specialist, association, and rendering.

Separate COCO-only transfer from matched vehicle adaptation. Our Co-DINO model is already vehicle-adapted, so comparing it against untouched RF-DETR/YOLO COCO weights cannot isolate architecture. Use the same train/development/evaluation partitions, disclose each model's resolution and pretraining, and select detector operating thresholds on development data at matched error budgets. Do not assume confidence values are calibrated equally across models.

No local RF-DETR or YOLO head-to-head has been executed yet. Published TensorRT/T4 detector timings cannot be compared directly with our PyTorch/MMCV/RTX-4090 full-pipeline FPS.
