"""Select vehicle teachers on calibration IDs; reserve final retrieval for reporting."""

import csv
from html import escape
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
from utils.core.metrics import cross_view_metrics
from utils.teacher.assets import sha256
from utils.tracking.common import ROOT
from .data import read, partitions, transform
from .model import build


class Images(Dataset):
    def __init__(self, records, conversion):
        self.records, self.conversion = records, conversion

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]['path']) as raw:
            return self.conversion(raw.convert('RGB'))


@torch.inference_mode()
def score(model, query, gallery, conversion, settings, pair_output=False):
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    for records in (query, gallery):
        loader = DataLoader(Images(records, conversion), batch_size=settings['evaluation']['batch_size'],
                            num_workers=settings['evaluation']['workers'])
        vectors = []
        for images in loader:
            result = model(images.to(device))
            if pair_output:
                result = result[1]
            vectors.append(F.normalize(result.float(), dim=1).cpu())
        outputs.append(torch.cat(vectors))
    correct, count, mean_ap = cross_view_metrics(outputs[0], [row['id'] for row in query],
                                                [row['camera'] for row in query], outputs[1],
                                                [row['id'] for row in gallery], [row['camera'] for row in gallery])
    return {'rank1': correct / count, 'map': mean_ap, 'queries': count, 'gallery': len(gallery)}


def candidates(settings, identities, include_finetuned=False):
    result = [('Compact warm-start ReID', 'compact', ROOT / settings['references']['compact']),
              ('MSINet VRU pretrained', 'msinet', ROOT / settings['references']['msinet_weights']),
              ('TransReID VeRi pretrained, camera-agnostic', 'transreid_pretrained', ROOT / settings['pretrained']['path'])]
    if include_finetuned:
        result.append(('TransReID VRAI best calibration checkpoint', 'transreid_finetuned',
                       ROOT / settings['training']['best_checkpoint']))
    return result


def load_candidate(kind, path, settings, identities):
    if kind == 'compact':
        from utils.benchmarking.models import _load_joint
        model = _load_joint(path)
        if model.preprocessing != 'vehicle_full_resize':
            raise ValueError('Expected the vehicle-resize compact reference')
        return model, transform(224, normalization='imagenet'), True
    if kind == 'msinet':
        import subprocess
        from utils.benchmarking.msinet_baseline import load_model
        revision = subprocess.check_output(['git', '-C', str(ROOT / settings['references']['msinet_source']),
                                            'rev-parse', 'HEAD'], text=True).strip()
        if revision != settings['references']['msinet_revision'] or sha256(path) != settings['references']['msinet_sha256']:
            raise ValueError('MSINet source/weights differ from the configured reference')
        model = load_model(ROOT / settings['references']['msinet_source'], path)
        return model, transform(256, normalization='imagenet'), False
    model = build(settings, identities, pretrained=kind == 'transreid_pretrained')
    if kind == 'transreid_finetuned':
        saved = torch.load(path, map_location='cpu', weights_only=False)
        from .training import validate
        validate(saved, settings)
        model.load_state_dict(saved['model'])
    return model, transform(settings['model']['image_size']), False


def write_report(settings, stage, records):
    output = ROOT / settings['evaluation']['output']
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'{stage}.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    headings = ''.join(f'<th>{escape(key)}</th>' for key in records[0])
    body = ''.join('<tr>' + ''.join(f'<td>{value:.2f}</td>' if isinstance(value, float)
                                  else f'<td>{escape(str(value))}</td>' for value in record.values()) + '</tr>'
                   for record in sorted(records, key=lambda row: -row['mAP']))
    (output / f'{stage}.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Vehicle ReID teachers</title>'
        '<style>body{font:16px system-ui;margin:40px;color:#172033}table{border-collapse:collapse}'
        'td,th{padding:12px;border-bottom:1px solid #ccd;text-align:right}</style>'
        f'<h1>Vehicle ReID teachers: {escape(stage)}</h1>'
        '<p>Same VRAI query/gallery per stage, other-camera matching and cosine similarity. Scores are percentages. '
        'Different pretrained datasets, model preprocessing and training budgets are reported in model names. '
        'Calibration chooses a teacher; final query/gallery are used only for reporting. '
        'This is a local split of VRAI training images, not the official VRAI test protocol. '
        'TransReID uses the released vehicle-trained backbone/JPM but disables source camera/view side embeddings.</p>'
        f'<table><tr>{headings}</tr>{body}</table>')


def run(final=False):
    settings = read()
    data = partitions(settings)
    identities = len({row['id'] for row in data['train']})
    include_finetuned = (ROOT / settings['training']['best_checkpoint']).is_file()
    if final and (include_finetuned or (ROOT / settings['training']['checkpoint']).is_file()):
        last = torch.load(ROOT / settings['training']['checkpoint'], map_location='cpu', weights_only=False)
        from .training import validate
        validate(last, settings)
        if not last['finished']:
            raise ValueError('Finish specialist training before final evaluation')
        last_run_id = last['run_id']
        del last
        if not include_finetuned:
            raise FileNotFoundError('Missing best specialist checkpoint')
        best = torch.load(ROOT / settings['training']['best_checkpoint'], map_location='cpu', weights_only=False)
        if best['run_id'] != last_run_id:
            raise ValueError('Specialist best/last checkpoints belong to different runs')
        del best
    stages = ['calibration', 'final'] if final else ['calibration']
    results = {}
    for stage in stages:
        query, gallery = ((data['calibration_query'], data['calibration_gallery']) if stage == 'calibration'
                          else (data['query'], data['gallery']))
        records = []
        for name, kind, path in candidates(settings, identities, include_finetuned):
            print(f'{stage}: evaluating {name}, {len(query)} queries / {len(gallery)} gallery', flush=True)
            model, conversion, pair = load_candidate(kind, path, settings, identities)
            model.cuda()
            metrics = score(model, query, gallery, conversion, settings, pair)
            record = {'model': name, 'kind': kind, 'checkpoint': str(path.relative_to(ROOT)),
                      'Rank1': metrics['rank1']*100, 'mAP': metrics['map']*100,
                      'queries': metrics['queries'], 'gallery': metrics['gallery'], 'checkpoint_sha256': sha256(path)}
            records.append(record)
            print(f'  Rank-1 {record["Rank1"]:.2f}%, mAP {record["mAP"]:.2f}%', flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
        write_report(settings, stage, records)
        results[stage] = records
        if stage == 'calibration':
            winner = max(records, key=lambda row: row['mAP'])
            destination = ROOT / 'checkpoints/reid_teacher' / ('selected.yaml' if final else 'provisional.yaml')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(yaml.safe_dump({'selection': 'calibration mAP only', 'candidate': winner,
                'finalized_selection': final, 'vrai_finetuning_complete': final and include_finetuned,
                'calibration_manifests': {key: sha256(ROOT / settings['data'][key])
                    for key in ('calibration_query', 'calibration_gallery')}}))
            print(f'Calibration winner: {winner["model"]}', flush=True)
    print(f'Reports: {ROOT / settings["evaluation"]["output"]}', flush=True)
    return results


def load_selected(device='cuda'):
    """Frozen, normalized identity teacher and its required image preprocessing."""
    from torch import nn
    settings = read()
    selection = yaml.safe_load((ROOT / 'checkpoints/reid_teacher/selected.yaml').read_text())
    if not selection['finalized_selection']:
        raise ValueError('Teacher selection is provisional')
    for key, digest in selection['calibration_manifests'].items():
        if sha256(ROOT / settings['data'][key]) != digest:
            raise ValueError('Calibration manifests differ from teacher selection')
    candidate = selection['candidate']
    path = ROOT / candidate['checkpoint']
    if sha256(path) != candidate['checkpoint_sha256']:
        raise ValueError('Selected teacher weights changed')
    identities = len({row['id'] for row in partitions(settings)['train']})
    network, conversion, pair = load_candidate(candidate['kind'], path, settings, identities)

    class FrozenTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network

        @torch.no_grad()
        def forward(self, images):
            self.network.eval()
            result = self.network(images)
            return F.normalize((result[1] if pair else result).float(), dim=1)

    model = FrozenTeacher().to(device).eval().requires_grad_(False)
    return model, conversion, selection
