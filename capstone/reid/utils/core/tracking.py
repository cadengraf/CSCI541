"""Motion and appearance association for one selected video object."""

import math

import torch.nn.functional as F


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1, area_left + area_right - intersection)


class TemporalSelector:
    """Associate one selected object across sampled frames without using track IDs."""

    def __init__(self, box, vector, frame):
        self.box = box
        self.vector = vector.detach().clone()
        self.frame = frame
        self.velocity = (0.0, 0.0)

    def choose(self, candidates, vectors, query_scores, frame):
        """Return a candidate index or None using motion, box, and appearance cues.

        Candidate track IDs are ignored. The caller uses them only to score
        the diagnostic video after a choice has been made.
        """
        elapsed = max(1, frame - self.frame)
        previous_center = _center(self.box)
        predicted = (previous_center[0] + self.velocity[0] * elapsed,
                     previous_center[1] + self.velocity[1] * elapsed)
        shift_x, shift_y = self.velocity[0] * elapsed, self.velocity[1] * elapsed
        predicted_box = (self.box[0] + shift_x, self.box[1] + shift_y,
                         self.box[2] + shift_x, self.box[3] + shift_y)
        diagonal = math.hypot(self.box[2] - self.box[0], self.box[3] - self.box[1])
        # Allow motion and modest camera shake, but reject distant lookalikes.
        scale = max(24.0, 1.5 * diagonal + 6.0 * math.sqrt(elapsed))
        temporal_scores = vectors @ self.vector
        choices = []
        for index, (_, _, box) in enumerate(candidates):
            center = _center(box)
            distance = math.dist(center, predicted)
            if distance > 2.5 * scale:
                continue
            proximity = math.exp(-0.5 * (distance / scale) ** 2)
            area = (box[2] - box[0]) * (box[3] - box[1])
            previous_area = (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])
            size_match = math.exp(-abs(math.log(area / previous_area)))
            # Spatial terms dominate nearby frames; embeddings help break ties.
            score = (0.30 * proximity + 0.35 * _iou(box, predicted_box)
                     + 0.20 * size_match + 0.10 * float(query_scores[index])
                     + 0.05 * float(temporal_scores[index]))
            choices.append((score, index))
        if not choices:
            return None
        best_score, selected = max(choices)
        if best_score < 0.5:
            return None
        new_center = _center(candidates[selected][2])
        observed_velocity = ((new_center[0] - previous_center[0]) / elapsed,
                             (new_center[1] - previous_center[1]) / elapsed)
        self.velocity = (0.5 * self.velocity[0] + 0.5 * observed_velocity[0],
                         0.5 * self.velocity[1] + 0.5 * observed_velocity[1])
        self.box = candidates[selected][2]
        self.vector = F.normalize(0.8 * self.vector + 0.2 * vectors[selected].detach(), dim=0)
        self.frame = frame
        return selected
