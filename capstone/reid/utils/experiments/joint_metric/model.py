"""Training-only metric substitution; state keys and inference model are unchanged."""
from utils.teacher.model import CoDINOJoint
from utils.experiments.appearance.losses import adasp


def metric_losses(original, embeddings, labels, margin, enabled, temperature):
    losses = original(embeddings, labels, margin)
    if enabled:
        # Preserve the original classification loss and common engine's loss key.
        # This key now holds AdaSP, not an additional triplet term.
        losses['loss_triplet'] = adasp(embeddings, labels, temperature)
    return losses


class MetricJoint(CoDINOJoint):
    def __init__(self, settings, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        self._vrai_batch = False
        original = self.reid.losses
        spec = settings['matched_reid']

        def losses(embeddings, labels, margin):
            return metric_losses(original, embeddings, labels, margin,
                self._vrai_batch and spec['vrai_metric']=='adasp', spec['metric_temperature'])

        self.reid.losses = losses

    def kitti_losses(self, *args, **kwargs):
        self._vrai_batch = False
        return super().kitti_losses(*args, **kwargs)

    def crop_embeddings(self, *args, **kwargs):
        self._vrai_batch = True
        return super().crop_embeddings(*args, **kwargs)
