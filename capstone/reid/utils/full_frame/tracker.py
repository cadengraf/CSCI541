"""Sequence-level vehicle association using motion, IoU, and ReID embeddings.

This is a simple Hungarian tracker, not an implementation of ByteTrack or MOTSNet.
It sees predictions only; ground truth is used exclusively by the evaluator.
"""

from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


@dataclass
class Track:
    identity: int
    box: torch.Tensor
    feature: torch.Tensor
    velocity: torch.Tensor
    last_frame: int


class VehicleTracker:
    def __init__(self, settings):
        self.settings = settings
        self.tracks = []
        self.next_identity = 1

    def update(self, prediction, frame):
        boxes = prediction['boxes'].cpu().float()
        features = F.normalize(prediction['embeddings'].cpu().float(), dim=1)
        if not torch.isfinite(boxes).all() or not torch.isfinite(features).all():
            raise ValueError('Tracker received non-finite boxes or embeddings')
        self.tracks = [track for track in self.tracks
                       if frame - track.last_frame <= self.settings['max_age']]
        identities = [-1] * len(boxes)
        if self.tracks and len(boxes):
            predicted_boxes = torch.stack([track.box + track.velocity * (frame - track.last_frame)
                                           for track in self.tracks])
            predicted_boxes[:, 2:] = torch.maximum(predicted_boxes[:, 2:], predicted_boxes[:, :2] + 1)
            overlaps = box_iou(predicted_boxes, boxes)
            similarities = torch.stack([track.feature for track in self.tracks]) @ features.T
            track_centers = (predicted_boxes[:, :2] + predicted_boxes[:, 2:]) / 2
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            scales = (predicted_boxes[:, 2:] - predicted_boxes[:, :2]).norm(dim=1).clamp_min(1)
            distance = torch.cdist(track_centers, centers) / scales[:, None]
            weight = self.settings['appearance_weight']
            costs = (1 - weight) * (1 - overlaps) + weight * (1 - similarities)
            allowed = overlaps >= self.settings['min_iou']
            if weight > 0:
                allowed |= similarities >= self.settings['min_cosine']
            allowed &= distance <= self.settings['max_center_distance']
            costs[~allowed] = 1e6
            rows, columns = linear_sum_assignment(costs.numpy())
            for row, column in zip(rows, columns):
                if costs[row, column] > self.settings['max_match_cost']:
                    continue
                track = self.tracks[row]
                elapsed = max(frame - track.last_frame, 1)
                track.velocity = 0.5 * track.velocity + 0.5 * (boxes[column] - track.box) / elapsed
                track.box = boxes[column].clone()
                momentum = self.settings['feature_momentum']
                track.feature = F.normalize(momentum * track.feature + (1 - momentum) * features[column], dim=0)
                track.last_frame = frame
                identities[column] = track.identity
        for index, identity in enumerate(identities):
            if identity != -1:
                continue
            self.tracks.append(Track(self.next_identity, boxes[index].clone(), features[index].clone(),
                                     torch.zeros(4), frame))
            identities[index] = self.next_identity
            self.next_identity += 1
        return identities


def disjoint_predictions(prediction, settings):
    """Assign overlap pixels to the highest-confidence mask, without GT access."""
    selected = torch.where(prediction['scores'].cpu() >= settings['score_threshold'])[0]
    selected = selected[torch.argsort(prediction['scores'].cpu()[selected], descending=True)]
    height, width = prediction['masks'].shape[-2:]
    occupied = torch.zeros((height, width), dtype=torch.bool)
    kept, masks = [], []
    for index in selected.tolist():
        mask = (prediction['masks'][index, 0].cpu() >= settings['mask_threshold']) & ~occupied
        if mask.sum().item() < settings['min_mask_pixels']:
            continue
        occupied |= mask
        kept.append(index)
        masks.append(mask)
    result = {key: prediction[key].cpu()[kept] for key in ('boxes', 'embeddings', 'scores')}
    result['masks'] = (torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.bool))
    return result
