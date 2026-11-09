"""Small, repeatable held-out KITTI MOTS diagnostic for the full-frame model."""

from pathlib import Path
from time import perf_counter
import csv

import torch
import yaml
from torchvision.ops import box_iou

from .kitti import KittiMOTSCars
from .model import FullFrameVehicleNet


def _match(prediction, target, score_threshold):
    chosen = torch.where(prediction['scores'] >= score_threshold)[0]
    if not len(chosen) or not len(target['boxes']):
        return [], len(chosen), len(target['boxes'])
    overlaps = box_iou(prediction['boxes'][chosen], target['boxes']).clone()
    matches = []
    while overlaps.numel():
        flat = overlaps.argmax().item()
        pred, truth = divmod(flat, overlaps.shape[1])
        if overlaps[pred, truth] < 0.5:
            break
        matches.append((chosen[pred].item(), truth))
        overlaps[pred, :] = -1
        overlaps[:, truth] = -1
    return matches, len(chosen) - len(matches), len(target['boxes']) - len(matches)


def _ap50_records(prediction, target):
    """Return ranked per-detection hits for one image at box IoU 0.5."""
    order = torch.argsort(prediction['scores'], descending=True)
    if not len(target['boxes']):
        return [(prediction['scores'][i].item(), False) for i in order]
    overlaps = box_iou(prediction['boxes'], target['boxes'])
    used = set()
    records = []
    for index in order.tolist():
        truth = int(overlaps[index].argmax())
        hit = overlaps[index, truth] >= 0.5 and truth not in used
        if hit:
            used.add(truth)
            overlaps[:, truth] = -1
        records.append((prediction['scores'][index].item(), bool(hit)))
    return records


def _ap50(records, ground_truth_count):
    if not ground_truth_count:
        return float('nan')
    ranked = sorted(records, key=lambda record: record[0], reverse=True)
    hits = 0
    precision_at_recall = []
    for rank, (_, hit) in enumerate(ranked, 1):
        hits += hit
        precision_at_recall.append((hits / ground_truth_count, hits / rank))
    return sum(max((precision for recall, precision in precision_at_recall if recall >= point),
                   default=0.0) for point in (i / 100 for i in range(101))) / 101


def evaluate(config_path, multiview=False):
    """Measure detection, mask, and adjacent-frame ID matching on held-out sequences.

    This is a diagnostic sample, not official KITTI AP/HOTA or cross-view ReID.
    """
    config_path = Path(config_path)
    settings = yaml.safe_load(config_path.read_text())['full_frame']
    root = Path(settings['data_root']).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    checkpoint = Path(settings['multiview']['checkpoint'] if multiview else settings['checkpoint']).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = config_path.parent / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Run `python train.py full-frame` first: {checkpoint}')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if set(saved['settings']['train_sequences']) & set(settings['val_sequences']):
        raise ValueError('Evaluation sequences overlap checkpoint training sequences')
    dataset = KittiMOTSCars(root, settings['val_sequences'])
    model = FullFrameVehicleNet(num_identities=saved['model']['id_classifier.weight'].shape[0],
                                pretrained=False, image_size=saved['settings']['image_size'],
                                architecture=saved['settings']['architecture'])
    model.load_state_dict(saved['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    requested = settings.get('eval_pairs')
    sample_count = min(int(requested), len(dataset)) if requested is not None else len(dataset)
    if sample_count < 1:
        raise ValueError('full_frame.eval_pairs must be positive')
    indices = torch.linspace(0, len(dataset) - 1, sample_count).round().long().tolist()
    counts = {threshold: [0, 0, 0] for threshold in (0.05, 0.5)}
    mask_ious = []
    correct = possible = 0
    ambiguous_correct = ambiguous_possible = 0
    scores = []
    ap_records = []
    ground_truth_count = 0
    inference_times = []
    with torch.inference_mode():
        for number, index in enumerate(indices, 1):
            pair = dataset[index]
            images = [frame[0].to(device) for frame in pair]
            targets = [frame[1] for frame in pair]
            start = perf_counter()
            predictions = [{key: value.cpu() for key, value in result.items()}
                           for result in model(images)]
            inference_times.append((perf_counter() - start) * 500)  # two frames, ms/frame
            # Each adjacent pair repeats one frame from its neighbors. Score only
            # the first frame of each pair so detection counts are not doubled.
            prediction, target = predictions[0], targets[0]
            scores.extend(prediction['scores'].tolist())
            ap_records.extend(_ap50_records(prediction, target))
            ground_truth_count += len(target['boxes'])
            for threshold in counts:
                matches, false_positives, misses = _match(prediction, target, threshold)
                count = counts[threshold]
                count[0] += len(matches)
                count[1] += false_positives
                count[2] += misses
                if threshold == 0.5:
                    for pred, truth in matches:
                        predicted_mask = prediction['masks'][pred, 0] >= 0.5
                        true_mask = target['masks'][truth].bool()
                        union = (predicted_mask | true_mask).sum().item()
                        mask_ious.append((predicted_mask & true_mask).sum().item() / union)
            left, _, _ = _match(predictions[0], targets[0], 0.5)
            right, _, _ = _match(predictions[1], targets[1], 0.5)
            right_ids = {targets[1]['track_ids'][truth].item(): pred for pred, truth in right}
            if right:
                candidates = torch.tensor([pred for pred, _ in right])
                for pred, truth in left:
                    identity = targets[0]['track_ids'][truth].item()
                    if identity not in right_ids:
                        continue
                    possible += 1
                    similarities = predictions[1]['embeddings'][candidates] @ predictions[0]['embeddings'][pred]
                    hit = int(candidates[similarities.argmax()].item() == right_ids[identity])
                    correct += hit
                    if len(candidates) > 1:
                        ambiguous_possible += 1
                        ambiguous_correct += hit
            if number % 25 == 0:
                print(f'Evaluated {number}/{sample_count} pairs', flush=True)
    print(f'Checkpoint epoch {saved["epoch"]}; {sample_count} of {len(dataset)} held-out adjacent pairs; device={device}')
    print(f'Box AP50 (101-point interpolation, unique first frames): {_ap50(ap_records, ground_truth_count):.3f}')
    for threshold, (true_positives, false_positives, misses) in counts.items():
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + misses, 1)
        print(f'Box IoU≥0.5, score≥{threshold}: precision={precision:.3f}, recall={recall:.3f} '
              f'(TP={true_positives}, FP={false_positives}, FN={misses})')
    print(f'Mean mask IoU for matched detections at score≥0.5: '
          f'{sum(mask_ious) / len(mask_ious):.3f} ({len(mask_ious)} masks)' if mask_ious
          else 'No detections matched at score≥0.5; mask IoU unavailable')
    print(f'Adjacent-frame ReID Rank-1 among matched detections: {correct / possible:.3f} '
          f'({correct}/{possible})' if possible else 'No eligible adjacent-frame ReID matches at score≥0.5')
    print(f'With at least two candidate cars: {ambiguous_correct / ambiguous_possible:.3f} '
          f'({ambiguous_correct}/{ambiguous_possible})' if ambiguous_possible
          else 'No adjacent-frame matches with at least two candidate cars')
    print(f'Detection score range: {min(scores):.3f}–{max(scores):.3f}' if scores
          else 'No detections returned')
    print(f'Median inference latency: {float(torch.tensor(inference_times).median()):.1f} ms/frame '
          f'({device}, 2-frame batches, includes output transfer, excludes image loading)')
    print('Local validation: AP50 is not COCO AP@[.50:.95], HOTA, or cross-camera ReID.')
    result = {'model': 'KITTI + VRAI' if multiview else 'KITTI only',
              'epoch': saved['epoch'], 'pairs': sample_count,
              'box_ap50': _ap50(ap_records, ground_truth_count),
              'matched_mask_iou': sum(mask_ious) / len(mask_ious) if mask_ious else None,
              'adjacent_rank1': correct / possible if possible else None,
              'multi_candidate_rank1': ambiguous_correct / ambiguous_possible if ambiguous_possible else None,
              'median_ms_per_frame': float(torch.tensor(inference_times).median()),
              'device': str(device)}
    for threshold, (tp, fp, fn) in counts.items():
        result[f'precision_score_{threshold}'] = tp / max(tp + fp, 1)
        result[f'recall_score_{threshold}'] = tp / max(tp + fn, 1)
    output = config_path.parent / 'metrics' / (
        'full_frame_multiview_kitti_local.csv' if multiview else 'full_frame_kitti_local.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial.csv')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    temporary.replace(output)
    print(f'Saved {output}', flush=True)
