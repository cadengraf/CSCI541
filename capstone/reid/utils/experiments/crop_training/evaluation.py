"""Cross-evaluate selected weights under both inference crop policies."""
import gc
import csv
import torch
import yaml
from utils.teacher.assets import sha256
from utils.teacher.data import build
from utils.teacher.evaluation import write_csv
from utils.teacher.model import CoDINOJoint
from utils.teacher.training import validate_checkpoint
from utils.core.metrics import cross_view_metrics
from utils.diagnostics.crops import extract
from .settings import ARMS, read


def cross_evaluate():
    records, provenance = [], {}
    for arm in ARMS:
        settings, root = read(arm)
        cfg = settings['matched_reid']
        output = root/cfg['output']
        source = root/settings['training']['best_checkpoint']
        saved = torch.load(source, map_location='cpu')
        kitti, vrai, partition, _ = build(settings, root)
        validate_checkpoint(saved, settings, kitti, vrai)
        manifests = {k: sha256(root/v) for k, v in settings['data'].items() if k.startswith('vrai_')}
        if saved['manifests'] != manifests:
            raise ValueError('VRAI manifests changed since training')
        provenance[arm] = dict(checkpoint=str(source), sha256=sha256(source), selected_epoch=saved['epoch'], manifests=manifests)
        model = CoDINOJoint(settings, root, len(kitti.identity_to_label)+len(vrai.identities), pretrained=False)
        model.load_state_dict(saved['model']); del saved
        model.cuda().eval()
        query, gallery = partition
        # Own-policy results already exist from calibration/KITTI evaluation.
        with (output/arm/'calibration.csv').open(newline='') as file:
            for row in csv.DictReader(file):
                records.append(dict(training_arm=arm, inference_policy=arm, pooling=row['pooling'],
                                    Rank1=float(row['Rank1']), mAP=float(row['mAP']),
                                    queries=int(row['queries']), gallery=int(row['gallery'])))
        for policy in ARMS:
            if policy == arm: continue
            chosen = read(policy)[0]['model']
            model.settings['model']['crop_geometry'] = chosen['crop_geometry']
            model.settings['model']['crop_size'] = chosen['crop_size']
            variant = dict(name=f'{arm} evaluated as {policy}', geometry=chosen['crop_geometry'], size=chosen['crop_size'])
            q, g = extract(model, query, variant), extract(model, gallery, variant)
            for pooling in ('mask', 'global'):
                correct, count, ap = cross_view_metrics(q[pooling], [r['id'] for r in query], [r['camera'] for r in query],
                    g[pooling], [r['id'] for r in gallery], [r['camera'] for r in gallery])
                row = dict(training_arm=arm, inference_policy=policy, pooling=pooling,
                           Rank1=100*correct/count, mAP=100*ap, queries=count, gallery=len(gallery))
                records.append(row)
                print(f'Cross-policy calibration: {row}', flush=True)
        write_csv(output/'cross_policy.csv', records)
        del model; gc.collect(); torch.cuda.empty_cache()
    (output/'cross_policy.yaml').write_text(yaml.safe_dump(provenance))
