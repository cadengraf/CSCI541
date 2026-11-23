"""Conservative open-set association. No ground truth or future frames enter here."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment


def unit(value):
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value)
    if value.ndim != 1 or not np.isfinite(value).all() or norm < 1e-8:
        raise ValueError('Expected a finite nonzero embedding vector')
    return value / norm


def iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = np.maximum(hi - lo, 0).prod()
    return float(intersection / max(1e-8, np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - intersection))


@dataclass
class Identity:
    identity: int
    references: list
    recent: list = field(default_factory=list)
    box: object = None
    frame: int = -1
    hits: int = 1
    confirmed: bool = False

    def similarity(self, feature):
        reference = max(float(value @ feature) for value in self.references)
        recent = max([float(value @ feature) for value in self.recent] or [reference])
        return .7 * reference + .3 * recent, reference


class Association:
    def __init__(self, settings):
        self.spec = settings
        for key in ('active_gap', 'recovery_hits', 'confirm_hits', 'tentative_age', 'memory_hits', 'recent_size'):
            if not isinstance(settings[key], int) or settings[key] < 1:
                raise ValueError(f'association.{key} must be a positive integer')
        for key in ('active_similarity', 'recovery_similarity', 'reference_floor', 'new_identity_max', 'memory_similarity'):
            if not -1 <= settings[key] <= 1:
                raise ValueError(f'association.{key} must be a cosine similarity in [-1, 1]')
        if settings['new_identity_max'] >= settings['active_similarity']:
            raise ValueError('new_identity_max must be below active_similarity')
        self.identities = {}
        self.next_id = 1
        self.pending = {}
        self.last_frame = -1

    def reset_view(self):
        """Preserve appearance but discard all coordinate and temporal evidence."""
        for item in self.identities.values():
            item.box, item.frame = None, -1
        self.pending.clear()
        self.last_frame = -1

    def update(self, boxes, features, reliable, frame):
        if frame <= self.last_frame:
            raise ValueError('Frames must increase; call reset_view for a new video/camera')
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) != len(features) or len(boxes) != len(reliable):
            raise ValueError('Observation fields have different lengths')
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError('Invalid boxes')
        features = [unit(value) for value in features]
        self.last_frame = frame
        spec = self.spec
        # Tentative clutter has no permanent identity evidence.
        self.identities = {key: item for key, item in self.identities.items()
                           if item.confirmed or frame-item.frame <= spec['tentative_age']}
        tracks = list(self.identities.values())
        n, m = len(tracks), len(boxes)
        scores = np.full((n, m), -1., dtype=np.float32)
        references = scores.copy()
        allowed = np.zeros((n, m), dtype=bool)
        active = []
        for row, item in enumerate(tracks):
            visible = item.box is not None and frame-item.frame <= spec['active_gap']
            active.append(visible)
            for column, (box, feature) in enumerate(zip(boxes, features)):
                score, reference = item.similarity(feature)
                scores[row, column], references[row, column] = score, reference
                if visible:
                    allowed[row, column] = (score >= spec['active_similarity'] and
                                            iou(item.box, box) >= spec['active_iou'])
                else:
                    allowed[row, column] = (item.confirmed and reliable[column] and
                        score >= spec['recovery_similarity'] and reference >= spec['reference_floor'])
        decisions = [dict(identity=None, status='unknown', similarity=0., margin=0., memory_update=False)
                     for _ in boxes]
        # Raw appearance competitors count even when spatially invalid. A repeated
        # failed match never becomes permission to relabel a plausible old object.
        for column in range(m):
            if n:
                decisions[column]['similarity'] = float(scores[:, column].max())
        if n and m:
            costs = np.where(allowed, 1-scores, 1e6)
            rows, columns = linear_sum_assignment(costs)
            next_pending = {}
            for row, column in zip(rows, columns):
                if not allowed[row, column]:
                    continue
                item = tracks[row]
                score = float(scores[row, column])
                other_tracks = np.delete(scores[:, column], row)
                other_objects = np.delete(scores[row], column)
                competitor = max(float(other_tracks.max()) if len(other_tracks) else -1.,
                                 float(other_objects.max()) if len(other_objects) else -1.)
                margin = score - competitor
                decision = decisions[column]
                decision.update(similarity=score, margin=margin)
                required = spec['active_margin'] if active[row] else spec['recovery_margin']
                if margin < required:
                    decision['status'] = 'ambiguous'
                    continue
                recovery = not active[row]
                if recovery:
                    previous = self.pending.get(item.identity)
                    count = (previous[0]+1 if previous and previous[1] == frame-1 and
                             iou(previous[2], boxes[column]) >= spec['pending_iou'] else 1)
                    next_pending[item.identity] = (count, frame, boxes[column].copy())
                    if count < spec['recovery_hits']:
                        decision['status'] = 'pending_recovery'
                        continue
                elapsed = frame-item.frame
                item.box, item.frame = boxes[column].copy(), frame
                item.hits = item.hits+1 if elapsed == 1 else 1
                item.confirmed |= item.hits >= spec['confirm_hits']
                # Recovery never immediately writes memory. Require a run of
                # clean continuous observations and agreement with frozen evidence.
                update = (not recovery and item.confirmed and reliable[column] and
                          item.hits >= spec['memory_hits'] and score >= spec['memory_similarity'] and
                          references[row, column] >= spec['reference_floor'] and margin >= spec['memory_margin'])
                if update:
                    item.recent.append(features[column].copy())
                    item.recent = item.recent[-spec['recent_size']:]
                decision.update(identity=item.identity, status=('recovered' if recovery else
                                'matched' if item.confirmed else 'tentative'), memory_update=bool(update))
            self.pending = next_pending
        else:
            self.pending.clear()
        for column, decision in enumerate(decisions):
            if decision['identity'] is not None:
                continue
            # An ambiguous/rejected old-identity candidate stays unknown.
            if not reliable[column] or (n and scores[:, column].max() >= spec['new_identity_max']):
                if decision['status'] == 'unknown':
                    decision['status'] = 'rejected'
                continue
            identity = self.next_id
            self.next_id += 1
            self.identities[identity] = Identity(identity, [features[column].copy()],
                box=boxes[column].copy(), frame=frame, confirmed=spec['confirm_hits'] <= 1)
            decision.update(identity=identity, status='new')
        return decisions
