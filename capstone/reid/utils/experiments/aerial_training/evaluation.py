"""Macro development retrieval plus separately calibrated rejection diagnostics."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.reid_teacher.evaluation import Images
from utils.reid_teacher.data import transform
from utils.experiments.appearance.evaluation import retrieval
from utils.vehicle_reid.operating_points import scores,calibrate,statistics


@torch.inference_mode()
def evaluate(model,data,settings,config):
    model.eval();device=next(model.parameters()).device;results={}
    for domain in ['vrai','vru']:
        stages={}
        for stage in ['fit','selection']:
            groups=[data[f'{domain}_{stage}_{kind}'] for kind in ['query','gallery']]
            vectors=[]
            for rows in groups:
                chunks=[]
                loader=DataLoader(Images(rows,transform(settings['model']['image_size'])),batch_size=32,num_workers=0)
                for images in loader:chunks.append(F.normalize(model(images.to(device)).float(),dim=1).cpu())
                vectors.append(torch.cat(chunks))
            ids=[[row['id'] for row in rows] for rows in groups]
            stages[stage]=dict(open=scores(*vectors,*ids),retrieval=retrieval(*vectors,*ids))
        point=calibrate(stages['fit']['open'],config['selection']['false_acceptance_budget'])
        observed=statistics(stages['selection']['open'],point['threshold'],point['margin'])
        results[domain]=dict(retrieval=stages['selection']['retrieval'],fit=point,open_set=observed)
    results['macro_map']=sum(results[name]['retrieval']['map'] for name in ['vrai','vru'])/2
    results['macro_rank1']=sum(results[name]['retrieval']['rank1'] for name in ['vrai','vru'])/2
    results['macro_known_acceptance']=sum(results[name]['open_set']['known_recovery'] for name in ['vrai','vru'])/2
    return results


def key(metrics):return metrics['macro_map'],metrics['macro_known_acceptance']
