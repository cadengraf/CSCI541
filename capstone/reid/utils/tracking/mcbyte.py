"""Pinned upstream McByte++ with cached vehicle features replacing person ReID."""

import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import torch
from torchvision.ops import box_iou
from .common import ROOT, vendor


class CachedVehicleReID:
    def is_enabled(self):
        return True

    def compute_feature_from_tlwh(self, image, tlwh):
        box = torch.tensor([tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]], dtype=torch.float32)[None]
        if not len(self.prediction['boxes']):
            return None
        ious = box_iou(box, self.prediction['boxes'])[0]
        index = int(ious.argmax())
        if ious[index] < .95:
            return None  # never borrow another detection's appearance
        return self.prediction['embeddings'][index].numpy().copy()


class McByte:
    def __init__(self, settings, frames, propagation):
        location = vendor(settings, 'mcbyte')
        # Import only upstream tracker modules, without YOLOX detector utilities.
        if 'yolox' not in sys.modules:
            package = ModuleType('yolox')
            package.__path__ = [str(location / 'yolox')]
            sys.modules['yolox'] = package
        from yolox.tracker import mcbyteplusplus_tracker__with_reid as upstream
        self.args = SimpleNamespace(**settings['mcbyte'])
        self.features = CachedVehicleReID()
        original_manager = upstream.ReIDManager
        upstream.ReIDManager = lambda **kwargs: self.features
        try:
            self.tracker = upstream.McBytePlusPlusTracker(self.args, save_folder=None, frame_rate=10, enable_logging=False)
        finally:
            upstream.ReIDManager = original_manager
        if self.tracker.gmc is not None:
            # Upstream initializes these guards only for ORB, but sparse flow
            # also calls _is_valid_affine. Supply the same guards for that path.
            for name, value in {'min_scale': .7, 'max_scale': 1.4, 'max_trans_frac': .25,
                                'last_good_H': np.eye(2, 3, dtype=np.float32)}.items():
                if not hasattr(self.tracker.gmc, name):
                    setattr(self.tracker.gmc, name, value)
        self.manager = None
        self.previous = ([], [], [], [])
        self.objects = {}
        if propagation:
            from .propagation import edge_manager
            self.manager = edge_manager(settings, frames, self.args)

    @torch.inference_mode()
    def update(self, prediction, image, frame):
        self.features.prediction = prediction
        propagated, mapping = None, {}
        if self.manager is not None and frame:
            propagated, mapping = self.manager.get_updated_masks(frame+1, *self.previous)
        detections = torch.cat((prediction['boxes'], prediction['scores'][:, None]), dim=1).numpy().copy()
        height, width = image.shape[:2]
        tracks, removed, new, _, _ = self.tracker.update(
            detections, [height, width], [height, width], propagated, mapping,
            image, 'none', dets_from_file=True)
        self.previous = ([track.last_det_tlwh for track in tracks], [track.track_id for track in tracks], new, removed)
        self.objects.update({track.track_id: track for track in tracks})
        indices, identities = [], []
        for track in tracks:
            box = torch.tensor(track.last_det_tlwh, dtype=torch.float32)
            box[2:] += box[:2]
            if not len(prediction['boxes']):
                continue
            overlaps = box_iou(box[None], prediction['boxes'])[0]
            index = int(overlaps.argmax())
            if overlaps[index] < .95:
                raise RuntimeError('Upstream track cannot be associated with its source detection')
            if index in indices:
                raise RuntimeError('Two tracks claimed one detection')
            indices.append(index)
            identities.append(track.track_id)
        return indices, identities

    def final_identity(self, internal):
        # Upstream demo buffers young tracks until their export IDs stabilize.
        track = self.objects[internal]
        return int(track.export_id if track.export_id is not None else internal)
