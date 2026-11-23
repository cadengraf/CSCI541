"""Exact measured local tracker, with trusted evidence saved for target profiles."""
import numpy as np
import torch
from utils.full_frame.tracker import VehicleTracker
from .association import unit
from .tracklets import TrackletAssociation


class ReferenceAssociation(TrackletAssociation):
    """Appearance memory records evidence but never alters baseline association."""
    def update(self, boxes, features, reliable, frame):
        if frame <= self.last_frame:
            raise ValueError('Frames must increase; call reset_view for a new video/camera')
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1,4)
        if len(boxes) != len(features) or len(boxes) != len(reliable):
            raise ValueError('Observation fields have different lengths')
        if not np.isfinite(boxes).all() or np.any(boxes[:,2:] <= boxes[:,:2]):
            raise ValueError('Invalid boxes')
        matrix = np.asarray(features,dtype=np.float32)
        if not len(boxes):
            matrix = np.empty((0,self.dimension or 0),dtype=np.float32)
        if matrix.ndim != 2 or (self.dimension is not None and matrix.shape[1] != self.dimension):
            raise ValueError('Embedding dimensions changed')
        if not np.isfinite(matrix).all() or (len(matrix) and np.any(np.linalg.norm(matrix,axis=1)<1e-8)):
            raise ValueError('Expected finite nonzero embeddings')
        previous = {track.identity:track.last_frame for track in self.local.tracks}
        # Use original input vectors, normalization, motion, and EMA exactly as the
        # measured VehicleTracker. Profile recording below cannot change its state.
        local_ids = self.local.update(dict(boxes=torch.from_numpy(boxes),embeddings=torch.from_numpy(matrix)),frame)
        self.dimension = matrix.shape[1] if len(matrix) else self.dimension
        self.last_frame = frame
        self.next_id = self.local.next_identity
        self.output_ids = {track.identity:track.identity for track in self.local.tracks}
        decisions = []
        for index,identity in enumerate(local_ids):
            kind = ('local_birth' if identity not in previous else
                    'local_reacquisition' if frame-previous[identity]>1 else 'local_continuation')
            decision = dict(identity=identity,tracking_id=identity,tracklet_id=identity,
                identity_scope='local',association_kind=kind,status='reference',
                similarity=0.,margin=0.,memory_update=False)
            feature = unit(matrix[index])
            if identity not in self.identities and reliable[index]:
                self._enroll(identity,boxes[index],feature,frame)
            elif identity in self.identities:
                recorded = self._touch(identity,boxes[index],feature,reliable[index],frame)
                decision.update(similarity=recorded['similarity'],margin=recorded['margin'],
                                memory_update=recorded['memory_update'])
            decisions.append(decision)
        return decisions

    def reset_view(self):
        super().reset_view()
        # A view reset starts fresh local IDs. Global profile matching is handled
        # by the explicitly selected experimental resolver, not this baseline.
        self.local.next_identity = self.next_id
