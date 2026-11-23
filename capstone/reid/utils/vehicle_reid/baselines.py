"""Pinned upstream algorithms with common detection-to-mask output adapters."""
from types import ModuleType, SimpleNamespace
import subprocess
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
from utils.tracking.common import ROOT
from utils.full_frame.tracker import VehicleTracker
from utils.full_frame.mots_evaluation import _numpy_compatibility
from .association import Association
from .tracklets import TrackletAssociation


def prepare_sources(config):
    for name, spec in config['source'].items():
        source = ROOT / spec['directory']
        if not source.exists():
            subprocess.run(['git', 'clone', spec['repository'], str(source)], check=True)
            subprocess.run(['git', '-C', str(source), 'checkout', spec['revision']], check=True)
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != spec['revision']:
            raise ValueError(f'{name} source revision changed')
        if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
            raise ValueError(f'{name} source must be unmodified')


def imports(config):
    """Avoid importing either upstream detector or installing its training stack."""
    path = ROOT / config['source']['bytetrack']['directory']
    if 'yolox' in sys.modules:
        if list(sys.modules['yolox'].__path__) != [str(path / 'yolox')]:
            raise RuntimeError('Another YOLOX tracker is loaded; run this benchmark in a fresh process')
    else:
        package = ModuleType('yolox')
        package.__path__ = [str(path / 'yolox')]
        sys.modules['yolox'] = package
    ocpath = ROOT / config['source']['ocsort']['directory'] / 'trackers'
    if 'trackers' not in sys.modules:
        package = ModuleType('trackers')
        package.__path__ = [str(ocpath)]
        sys.modules['trackers'] = package
    from yolox.tracker.byte_tracker import BYTETracker, STrack
    # Observe the upstream measurement passed into each Kalman update, rather
    # than rematching its smoothed state to a mask. This changes no association,
    # identity, threshold, or motion state and preserves exact source masks.
    if not getattr(STrack, '_vehicle_mask_observer', False):
        from functools import wraps
        def observe(function):
            @wraps(function)
            def wrapped(self, new_track, *args, **kwargs):
                self.vehicle_observation = new_track.tlbr.copy()
                return function(self, new_track, *args, **kwargs)
            return wrapped
        STrack.update = observe(STrack.update)
        STrack.re_activate = observe(STrack.re_activate)
        STrack._vehicle_mask_observer = True
    from trackers.ocsort_tracker.ocsort import OCSort
    return BYTETracker, OCSort


class Method:
    def __init__(self, name, config, pipeline, legacy):
        self.name = name
        self.mapping_drops = 0
        if name in ('bytetrack', 'ocsort'):
            byte, oc = imports(config)
            if name == 'bytetrack':
                from yolox.tracker.basetrack import BaseTrack
                BaseTrack._count = 0
                self.algorithm = byte(SimpleNamespace(**config['bytetrack']), frame_rate=10)
            else:
                self.algorithm = oc(**config['ocsort'])
        elif name.startswith('legacy_'):
            self.algorithm = VehicleTracker(legacy)
        elif name.startswith('tracklet_'):
            self.algorithm = TrackletAssociation(pipeline['association'])
        elif name.startswith('memory_'):
            spec = dict(pipeline['association'])
            if name.endswith('_unguarded'):
                spec.update(memory_hits=1, memory_similarity=-1., memory_margin=-2., reference_floor=-1.)
            self.algorithm = Association(spec)
        else:
            raise ValueError(f'Unknown benchmark method {name}')

    def update(self, raw, observations, reliable, frame):
        if self.name in ('bytetrack', 'ocsort'):
            boxes = raw['boxes'].float()
            candidates = torch.cat((boxes, raw['scores'][:, None]), dim=1).numpy().copy()
            shape = raw['masks'].shape[-2:]
            with _numpy_compatibility():
                output = self.algorithm.update(candidates, shape, shape)
            if self.name == 'bytetrack':
                returned = np.asarray([np.r_[getattr(track, "vehicle_observation", track.tlbr), track.track_id] for track in output], dtype=np.float32).reshape(-1, 5)
            else:
                returned = output
            emitted = []
            if len(returned) and len(boxes):
                overlaps = box_iou(torch.as_tensor(returned[:, :4], dtype=torch.float32), boxes).numpy()
                rows, columns = linear_sum_assignment(1-overlaps)
                for row, column in zip(rows, columns):
                    if overlaps[row, column] >= .5:
                        emitted.append((int(column), dict(identity=int(returned[row, 4]), status='upstream')))
                self.mapping_drops += len(returned)-len(emitted)
                if len(returned) != len(emitted):
                    raise RuntimeError(f'{self.name}: output cannot be mapped to its current source detection')
            else:
                self.mapping_drops += len(returned)
            return emitted
        indices = observations['indices'].tolist()
        boxes = observations['boxes']
        features = (raw['embeddings'][indices] if self.name.endswith('_joint') else observations['embeddings'])
        if self.name.startswith('legacy_'):
            identities = self.algorithm.update(dict(boxes=boxes, embeddings=features), frame)
            decisions = [dict(identity=identity, status='legacy') for identity in identities]
        else:
            quality = [True]*len(indices) if self.name.endswith('_unguarded') else reliable
            decisions = self.algorithm.update(boxes.numpy(), features.numpy(), quality, frame)
        # Unknown observations get unique one-frame identities for MOT evaluation.
        # This preserves their detection contribution while penalizing fragmentation.
        # They remain unknown for recovery diagnostics, never accepted matches.
        return list(zip(indices, decisions))
