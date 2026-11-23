"""Fit rejection on one calibration identity subset; select on the other."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.reid_teacher.evaluation import Images
from utils.reid_teacher.data import transform
from utils.vehicle_reid.operating_points import scores,calibrate,statistics


def retrieval(query,gallery,query_ids,gallery_ids):
    values=(query@gallery.T).numpy()
    ranking=np.argsort(-values,axis=1,kind='stable')
    matches=np.asarray(gallery_ids)[ranking]==np.asarray(query_ids)[:,None]
    counts=matches.sum(1)
    if not np.all(counts>0):raise ValueError('Query has no gallery positive')
    ap=(np.cumsum(matches,axis=1)/np.arange(1,len(gallery_ids)+1)*matches).sum(1)/counts
    return dict(rank1=float(matches[:,0].mean()),map=float(ap.mean()),queries=len(query_ids),gallery=len(gallery_ids))


@torch.inference_mode()
def evaluate(model,data,settings,experiment):
    model.eval();device=next(model.parameters()).device
    records=[data['calibration_query'],data['calibration_gallery']]
    vectors=[]
    for group in records:
        loader=DataLoader(Images(group,transform(settings['model']['image_size'])),
            batch_size=settings['evaluation']['batch_size'],num_workers=0)
        chunks=[]
        for images in loader:
            chunks.append(F.normalize(model(images.to(device)).float(),dim=1).cpu())
        vectors.append(torch.cat(chunks))
    query_ids=[row['id'] for row in records[0]];gallery_ids=[row['id'] for row in records[1]]
    identities=sorted(set(query_ids))
    fit=set(identities[::2]);selection=set(identities[1::2])
    if fit&selection:raise ValueError('Calibration fit/selection identities overlap')
    stages={}
    for name,ids in [('fit',fit),('selection',selection)]:
        qi=[i for i,value in enumerate(query_ids) if value in ids]
        gi=[i for i,value in enumerate(gallery_ids) if value in ids]
        stages[name]=scores(vectors[0][qi],vectors[1][gi],[query_ids[i] for i in qi],[gallery_ids[i] for i in gi])
    budget=experiment['selection']['false_acceptance_budget']
    point=calibrate(stages['fit'],budget)
    measured=statistics(stages['selection'],point['threshold'],point['margin'])
    eligible=measured['unknown_far']<=budget and measured['wrong_known_rate']<=budget
    return dict(closed_set=retrieval(*vectors,query_ids,gallery_ids),fit=point,selection=measured,
                eligible=bool(eligible),fit_identities=len(fit),selection_identities=len(selection))


def key(metrics):
    return (int(metrics['eligible']),metrics['selection']['known_recovery'],
            -metrics['selection']['wrong_known_rate'],-metrics['selection']['unknown_far'],metrics['closed_set']['map'])
