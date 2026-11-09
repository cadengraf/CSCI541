"""Load released mask propagators; no detector or identity-head replacement."""

from functools import lru_cache
import sys
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from .common import ROOT, vendor


class LazyFrames:
    """Read original PNGs on demand instead of duplicating videos as JPEGs."""
    def __init__(self, frames, size):
        self.frames, self.size = frames, size

    def __len__(self):
        return len(self.frames)

    @lru_cache(maxsize=2)
    def __getitem__(self, index):
        with Image.open(self.frames[index]) as raw:
            image = pil_to_tensor(raw.convert('RGB').resize((self.size, self.size))).float() / 255
        return (image - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def edge_manager(settings, frames, args):
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    mcbyte = ROOT / settings['mcbyte_source']
    sys.path.insert(0, str(mcbyte / 'mask_propagation/EdgeTAM'))
    from sam2.build_sam import build_sam2_video_predictor
    import sam2.sam2_video_predictor as video
    from mask_propagation import mask_manager__edgetam as upstream

    def lazy_loader(**kwargs):
        with Image.open(frames[0]) as raw:
            width, height = raw.size
        return LazyFrames(frames, kwargs['image_size']), height, width

    # Upstream loader accepts JPEG folders; our frame adapter keeps KITTI PNGs.
    video.load_video_frames = lazy_loader
    # sam2 initializes Hydra only on its first import. Python caches that import,
    # so clearing Hydra between sequences requires explicit reinitialization.
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module='sam2', version_base='1.2'):
        predictor = build_sam2_video_predictor('configs/edgetam.yaml', str(ROOT / settings['edgetam_weights']),
                                               apply_postprocessing=False)
    initialize = predictor.init_state
    predictor.init_state = lambda **kwargs: initialize(**kwargs, offload_video_to_cpu=True, offload_state_to_cpu=True)
    original = upstream.build_sam2_video_predictor
    upstream.build_sam2_video_predictor = lambda *args, **kwargs: predictor
    try:
        manager = upstream.MaskManager(str(frames[0].parent), args)
    finally:
        upstream.build_sam2_video_predictor = original
    manager.tracklet_mask_dict = {}
    initialize_masks = manager.initialize_first_masks

    def empty_safe(*args, **kwargs):
        # Upstream returns bare None when initial box prompts are all rejected.
        result = initialize_masks(*args, **kwargs)
        return (None, None) if result is None else result

    manager.initialize_first_masks = empty_safe
    return manager


def cutie_processor(settings):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    source = vendor(settings, 'cutie')
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base='1.3', config_dir=str(source / 'cutie/config')):
        cfg = compose(config_name='eval_config')
    cfg.use_long_term = True
    cfg.mem_every = 5
    from cutie.model.cutie import CUTIE
    from cutie.inference.inference_core import InferenceCore
    model = CUTIE(cfg).cuda().eval()
    weights = torch.load(ROOT / settings['cutie_weights'], map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    processor = InferenceCore(model, cfg=cfg)
    processor.max_internal_size = settings['selected']['cutie_max_internal_size']
    return processor
