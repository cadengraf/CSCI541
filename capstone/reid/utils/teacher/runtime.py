"""Entry point executed in the isolated teacher environment."""

import argparse
import torch


def check():
    from .settings import read
    from .data import build
    from .model import CoDINOJoint
    from .training import optimizer_for, total_loss, backward_step
    settings, root = read()
    if not torch.cuda.is_available():
        raise RuntimeError('The teacher requires CUDA')
    kitti, vrai, _, _ = build(settings, root)
    offset = len(kitti.identity_to_label)
    model = CoDINOJoint(settings, root, offset + len(vrai.identities)).cuda()
    from utils.full_frame.training import _pair_collate
    crowded = max(range(len(kitti)), key=lambda index: sum(len(kitti.annotations[key]) for key in kitti.pairs[index]))
    images, targets = _pair_collate([kitti[crowded]])
    print(f'Checking crowded KITTI pair with {[len(target["boxes"]) for target in targets]} cars', flush=True)
    images = [image.cuda() for image in images]
    model.eval()
    with torch.no_grad():
        prediction = model(images[:1])[0]
    print(f'Pretrained inference: {len(prediction["boxes"])} vehicles; '
          f'masks {tuple(prediction["masks"].shape)}; embeddings {tuple(prediction["embeddings"].shape)}', flush=True)
    del prediction
    optimizer = optimizer_for(model, settings)
    scaler = torch.cuda.amp.GradScaler(enabled=settings['training']['mixed_precision'], init_scale=1024)
    for epoch in (0, settings['training']['frozen_backbone_epochs']):
        print(f'Checking {model.phase(epoch)}', flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.amp.autocast(enabled=settings['training']['mixed_precision']):
            losses = model.kitti_losses(images, targets, augment=False)
            loss = total_loss(losses, settings['training']['reid_weight'])
        print({key: float(value.detach()) for key, value in losses.items() if key in ('loss_identity', 'loss_triplet')}, flush=True)
        updated = backward_step(model, optimizer, scaler, loss)
        if not updated:
            raise RuntimeError('Check encountered AMP overflow; lower initial loss scale before training')
        assert model.reid.classifier.grad is not None
        if epoch:
            assert any(parameter.grad is not None for parameter in model.detector.backbone.parameters())
        print(f'Joint step passed: loss {loss.item():.3f}, peak CUDA {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
        del losses, loss
    import random
    crops, labels = vrai.sample(random.Random(42), settings['training']['identities_per_batch'])
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=settings['training']['mixed_precision']):
        embeddings = model.crop_embeddings([image.cuda() for image in crops])
        loss = total_loss(model.reid.losses(embeddings, labels.cuda() + offset,
                                           settings['training']['triplet_margin']), 1)
    backward_step(model, optimizer, scaler, loss)
    print(f'Cross-view identity step passed: loss {loss.item():.3f}', flush=True)
    del embeddings, loss, crops
    model.eval()
    from pathlib import Path
    import tempfile
    from utils.full_frame.mots_benchmark import infer_sequence
    from utils.full_frame.mots_evaluation import evaluate
    from .settings import resolve
    sequence = f'{settings["data"]["val_sequences"][0]:04d}'
    data_root = resolve(root, settings['data']['kitti_root'])
    with tempfile.TemporaryDirectory(prefix='teacher_check_') as directory:
        folder = Path(directory)
        truth = folder / 'truth/instances_txt'
        truth.mkdir(parents=True)
        lines = (data_root / 'instances_txt' / f'{sequence}.txt').read_text().splitlines()
        (truth / f'{sequence}.txt').write_text('\n'.join(line for line in lines if int(line.split()[0]) < 2) + '\n')
        infer_sequence(model, data_root / 'training/image_02' / sequence,
                       folder / 'trackers/check/data' / f'{sequence}.txt', settings['tracking'],
                       torch.device('cuda'), frame_limit=2)
        evaluate(folder / 'truth', folder / 'trackers', ['check'], {sequence: 2}, folder / 'metrics', print_results=False)
    print('Mask export, tracking, and official TrackEval check passed', flush=True)
    print('Checks passed. No training checkpoint was written.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'train', 'eval', 'tracking-cache', 'tracking-cache-check', 'tracking-cache-selected',
        'crop-train-aspect_384', 'crop-train-square_256', 'crop-train-aspect_384-check', 'crop-train-square_256-check',
        'crop-train-aspect_384-eval', 'crop-train-square_256-eval', 'crop-train-cross-eval', 'roi-reid-raw_roi', 'roi-reid-reid_adapter', 'roi-reid-raw_roi-check', 'roi-reid-reid_adapter-check',
        'roi-reid-raw_roi-eval', 'roi-reid-reid_adapter-eval', 'reid-balance-reid_only', 'reid-balance-joint', 'reid-balance-reid_only-check', 'reid-balance-joint-check',
        'reid-balance-reid_only-eval', 'reid-balance-joint-eval', 'reid-overfit', 'reid-diagnostics', 'crop-diagnostics', 'crop-validation', 'joint-distill', 'joint-distill-check', 'joint-distill-control', 'joint-distill-eval', 'joint-distill-control-eval'))
    mode = parser.parse_args().mode
    if mode == 'crop-train-cross-eval':
        from utils.experiments.crop_training.evaluation import cross_evaluate
        cross_evaluate()
    elif mode.startswith('crop-train-'):
        from utils.experiments.crop_training.settings import read as crop_settings
        arm = mode.removeprefix('crop-train-').split('-')[0]
        if mode.endswith('-eval'):
            from .evaluation import run
            run(experiment=crop_settings(arm)[0], calibration_only=True)
        else:
            from utils.experiments.reid_balance.training import train as compare
            compare(arm, check=mode.endswith('-check'), settings_reader=crop_settings)
    elif mode in ('crop-diagnostics', 'crop-validation'):
        from utils.diagnostics.crops import run
        run(full=mode == 'crop-validation')
    elif mode.startswith('roi-reid-'):
        from utils.experiments.roi_reid.settings import read as roi_settings
        arm = mode.removeprefix('roi-reid-').split('-')[0]
        if mode.endswith('-eval'):
            from .evaluation import run
            run(experiment=roi_settings(arm)[0], calibration_only=True)
        else:
            from utils.experiments.reid_balance.training import train as compare
            compare(arm, check=mode.endswith('-check'), settings_reader=roi_settings)
    elif mode.startswith('reid-balance-'):
        arm = mode.removeprefix('reid-balance-').split('-')[0]
        if mode.endswith('-eval'):
            from utils.experiments.reid_balance.settings import read as balance_settings
            from .evaluation import run
            run(experiment=balance_settings(arm)[0], calibration_only=True)
        else:
            from utils.experiments.reid_balance.training import train as balance
            balance(arm, check=mode.endswith('-check'))
    elif mode == 'reid-overfit':
        from utils.diagnostics.network import overfit_network
        overfit_network()
    elif mode == 'reid-diagnostics':
        from utils.diagnostics.reid import run
        run()
    elif mode.startswith('joint-distill'):
        if mode.endswith('-eval'):
            from utils.distillation.settings import read as distilled_settings
            from .evaluation import run
            run(experiment=distilled_settings(control='control' in mode)[0])
        else:
            from utils.distillation.training import train as distill
            distill(check=mode.endswith('-check'), control='control' in mode)
    elif mode.startswith('tracking-cache'):
        from utils.tracking.cache import export
        export(check=mode.endswith('-check'), selected=mode.endswith('-selected'))
    elif mode == 'check':
        check()
    elif mode == 'train':
        from .training import train
        train()
    else:
        from .evaluation import run
        run()


if __name__ == '__main__':
    main()
