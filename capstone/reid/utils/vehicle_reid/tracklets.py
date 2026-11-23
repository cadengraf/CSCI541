"""Local continuity and persistent identity recovery are independent decisions."""
from copy import deepcopy
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from utils.full_frame.tracker import VehicleTracker
from .association import Association, Identity, unit


LOCAL_DEFAULTS = dict(max_age=15, appearance_weight=.5, min_iou=.05, min_cosine=.7,
                      max_center_distance=2., max_match_cost=.8, feature_momentum=.9)
LOCAL_ID_OFFSET = 1_000_000_000


def tracking_identity(decision):
    """Stable local output without claiming an uncertain global reconnection."""
    if 'tracking_id' in decision:
        return decision['tracking_id']
    if decision['identity'] is not None:
        return decision['identity']
    local = decision.get('tracklet_id')
    return LOCAL_ID_OFFSET + local if local is not None else None


class TrackletAssociation:
    def __init__(self, settings):
        # Keep the v1 threshold validation and its frozen appearance representation.
        Association(settings)
        self.spec = deepcopy(settings)
        self.local_settings = {**LOCAL_DEFAULTS, **settings.get('local', {})}
        if self.local_settings['max_age'] < 1:
            raise ValueError('association.local.max_age must be positive')
        self.local = VehicleTracker(self.local_settings)
        self.identities = {}
        self.bindings = {}  # Local tracklet -> persistent identity, one owner each.
        self.output_ids = {}  # Reserve an output ID at birth, before global enrollment.
        self.pending = {}   # Local tracklet -> (candidate identity, count, frame).
        self.next_id = 1
        self.last_frame = -1
        self.dimension = None

    def reset_view(self):
        self.local = VehicleTracker(self.local_settings)
        self.bindings.clear()
        self.output_ids.clear()
        self.pending.clear()
        self.last_frame = -1
        for item in self.identities.values():
            item.box, item.frame = None, -1
            item.hits = 0
            item.clean_hits = 0

    def _observe_tracklets(self,local_ids,boxes,features,reliable,frame):
        """Optional appearance-evidence observer; default inference is unchanged."""
        pass

    def _recovery_similarity(self,item,feature,local,frame):
        return item.similarity(feature)

    def _recovery_ready(self,item,local,frame):
        return True

    def _memory_write_allowed(self,item,frame):
        return True

    def _margin(self, identity, feature):
        score, reference = self.identities[identity].similarity(feature)
        competitor = max((item.similarity(feature)[0] for key, item in self.identities.items()
                          if key != identity), default=-1.)
        return score, reference, score-competitor

    def _touch(self, identity, box, feature, reliable, frame, recovery=False):
        item = self.identities[identity]
        score, reference, margin = self._margin(identity, feature)
        gap = frame-item.frame
        continuous = not recovery and gap == 1
        kind = 'global_recovery' if recovery else 'local_reacquisition' if gap>1 else 'local_continuation'
        item.hits = item.hits+1 if continuous else 1
        item.clean_hits = (getattr(item, 'clean_hits', 1)+1 if continuous else 1) if reliable else 0
        item.confirmed |= item.clean_hits >= self.spec['confirm_hits']
        update = (not recovery and item.confirmed and reliable and item.clean_hits >= self.spec['memory_hits'] and
                  score >= self.spec['memory_similarity'] and reference >= self.spec['reference_floor'] and
                  margin >= self.spec['memory_margin'] and self._memory_write_allowed(item,frame))
        if update:
            item.recent.append(feature.copy())
            item.recent = item.recent[-self.spec['recent_size']:]
        item.box, item.frame = box.copy(), frame
        return dict(identity=identity, association_kind=kind, status='recovered' if recovery else 'matched' if item.confirmed else 'tentative',
                    similarity=score, margin=margin, memory_update=bool(update))

    def _enroll(self, local, box, feature, frame):
        identity = self.output_ids[local]
        if identity in self.identities:
            raise RuntimeError('Enrollment would overwrite an existing global identity')
        item = Identity(identity, [feature.copy()], box=box.copy(), frame=frame,
                        confirmed=self.spec['confirm_hits'] <= 1)
        item.clean_hits = 1
        self.identities[identity] = item
        self.bindings[local] = identity
        return identity

    def update(self, boxes, features, reliable, frame):
        if frame <= self.last_frame:
            raise ValueError('Frames must increase; call reset_view for a new video/camera')
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) != len(features) or len(boxes) != len(reliable):
            raise ValueError('Observation fields have different lengths')
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError('Invalid boxes')
        features = [unit(feature) for feature in features]
        dimension = len(features[0]) if features else self.dimension
        if any(len(feature) != dimension for feature in features) or (self.dimension is not None and dimension != self.dimension):
            raise ValueError('Embedding dimensions changed')
        self.dimension = dimension
        matrix = np.stack(features) if features else np.empty((0, dimension or 0), dtype=np.float32)
        previous_features = {track.identity: track.feature.clone() for track in self.local.tracks}
        local_ids = self.local.update(dict(boxes=torch.from_numpy(boxes), embeddings=torch.from_numpy(matrix)), frame)
        unreliable = {local for local, good in zip(local_ids, reliable) if not good}
        for track in self.local.tracks:
            if track.identity in unreliable and track.identity in previous_features:
                track.feature = previous_features[track.identity]
        self._observe_tracklets(local_ids,boxes,features,reliable,frame)
        self.last_frame = frame
        live_local = {item.identity for item in self.local.tracks}
        self.identities = {key: item for key, item in self.identities.items()
                           if item.confirmed or frame-item.frame <= self.spec['tentative_age']}
        self.bindings = {local: identity for local, identity in self.bindings.items()
                         if local in live_local and identity in self.identities}
        self.output_ids = {local: identity for local, identity in self.output_ids.items() if local in live_local}
        for local in local_ids:
            if local not in self.output_ids:
                if self.next_id >= LOCAL_ID_OFFSET:
                    raise RuntimeError('Tracking identity namespace exhausted')
                self.output_ids[local] = self.next_id
                self.next_id += 1
        visible_identities = {self.bindings[local] for local in local_ids if local in self.bindings}
        if len(visible_identities) != sum(local in self.bindings for local in local_ids):
            raise RuntimeError('Two visible tracklets own the same persistent identity')
        decisions = [dict(identity=None, tracklet_id=local, status='local_only', similarity=0., margin=0., memory_update=False)
                     for local in local_ids]
        unresolved = []
        for column, local in enumerate(local_ids):
            if local in self.bindings:
                # Global ambiguity NEVER tears down already established local continuity.
                decisions[column].update(self._touch(self.bindings[local], boxes[column], features[column],
                                                      reliable[column], frame))
            else:
                unresolved.append(column)
        # Only absent persistent identities can be proposed to new local tracklets.
        lost = [item for key, item in self.identities.items() if key not in visible_identities]
        scores = np.full((len(lost), len(unresolved)), -1., dtype=np.float32)
        references = scores.copy()
        allowed = np.zeros(scores.shape, dtype=bool)
        for row, item in enumerate(lost):
            for position, column in enumerate(unresolved):
                score, reference = self._recovery_similarity(item,features[column],local_ids[column],frame)
                scores[row, position], references[row, position] = score, reference
                allowed[row, position] = (item.confirmed and reliable[column] and
                    self._recovery_ready(item,local_ids[column],frame) and
                    score >= self.spec['recovery_similarity'] and reference >= self.spec['reference_floor'])
        next_pending = {}
        if len(lost) and len(unresolved):
            rows, positions = linear_sum_assignment(np.where(allowed, 1-scores, 1e6))
            for row, position in zip(rows, positions):
                if not allowed[row, position]:
                    continue
                column = unresolved[position]
                item, local = lost[row], local_ids[column]
                score = float(scores[row, position])
                other_identities = np.delete(scores[:, position], row)
                other_tracklets = np.delete(scores[row], position)
                competitor = max(float(other_identities.max()) if len(other_identities) else -1.,
                                 float(other_tracklets.max()) if len(other_tracklets) else -1.)
                margin = score-competitor
                decisions[column].update(similarity=score, margin=margin)
                if margin < self.spec['recovery_margin']:
                    decisions[column]['status'] = 'ambiguous'
                    continue
                previous = self.pending.get(local)
                count = previous[1]+1 if previous and previous[0] == item.identity and previous[2] == frame-1 else 1
                next_pending[local] = (item.identity, count, frame)
                if count < self.spec['recovery_hits']:
                    decisions[column]['status'] = 'pending_recovery'
                    continue
                # Transfer ownership at this frame only. Previous output is never rewritten.
                previous_owners = [key for key, value in self.bindings.items() if value == item.identity and key != local]
                for owner in previous_owners:
                    self.output_ids.pop(owner, None)
                self.bindings = {key: value for key, value in self.bindings.items() if value != item.identity}
                self.bindings[local] = item.identity
                self.output_ids[local] = item.identity
                decisions[column].update(self._touch(item.identity, boxes[column], features[column], True, frame, recovery=True))
                decisions[column]['margin'] = margin
        self.pending = next_pending
        for position, column in enumerate(unresolved):
            if decisions[column]['identity'] is not None:
                continue
            maximum = float(scores[:, position].max()) if len(lost) else -1.
            decisions[column]['similarity'] = max(decisions[column]['similarity'], maximum)
            if not reliable[column] or maximum >= self.spec['new_identity_max']:
                if decisions[column]['status'] == 'local_only':
                    decisions[column]['status'] = 'rejected'
                continue
            identity = self._enroll(local_ids[column], boxes[column], features[column], frame)
            decisions[column].update(identity=identity, status='new')
        for decision, local in zip(decisions, local_ids):
            decision['tracking_id'] = self.output_ids[local]
        if len({decision['tracking_id'] for decision in decisions}) != len(decisions):
            raise RuntimeError('Duplicate output identity in one frame')
        return decisions


def build(settings):
    engine = settings.get('engine', 'memory_v1')
    if engine == 'aggregated':
        from .aggregation import TrackletEvidenceAssociation
        return TrackletEvidenceAssociation(settings)
    if engine == 'reference':
        from .reference import ReferenceAssociation
        return ReferenceAssociation(settings)
    if engine == 'tracklets':
        return TrackletAssociation(settings)
    if engine == 'memory_v1':
        return Association(settings)
    raise ValueError(f'Unknown association engine: {engine}')
