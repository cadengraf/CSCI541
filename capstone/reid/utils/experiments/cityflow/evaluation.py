"""Supplied official evaluator, with explicit API-only compatibility adapters."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
import numpy as np
import pandas as pd
import yaml
from utils.tracking.common import ROOT,load


@contextmanager
def compatibility():
    added=not hasattr(np,'asfarray')
    if added:np.asfarray=lambda value,dtype=float:np.asarray(value,dtype=dtype)
    try:yield
    finally:
        if added:delattr(np,'asfarray')


def official(path):
    tree=ast.parse(Path(path).read_text());removed=0
    for node in list(tree.body):
        if isinstance(node,ast.Import) and any(alias.name=='pytrec_eval' for alias in node.names):
            if any(isinstance(n,ast.Name) and n.id=='trec' and isinstance(n.ctx,ast.Load) for n in ast.walk(tree)):
                raise ValueError('pytrec_eval is actually used; install it rather than removing its import')
            tree.body.remove(node);removed+=1
    if removed!=1:raise ValueError('Unexpected official evaluator import layout')
    module=ModuleType('cityflow_supplied_evaluator');module.__file__=str(path)
    exec(compile(tree,str(path),'exec'),module.__dict__)
    return module


def score(config,prediction,gt_path=None,roi_root=None):
    root=Path(config['dataset_root']);evaluator=official(root/'eval/eval.py')
    ground=evaluator.readData(str(gt_path or root/'eval/ground_truth_validation.txt'))
    predicted=evaluator.readData(str(prediction))
    if config.get('check_frames'):
        selection=load(ROOT/config['output']/'selection.pt');cameras={int(cam[1:]) for scene in selection['scenes'].values() for cam in scene}
        ground=ground[ground.CameraId.isin(cameras)&(ground.FrameId<=config['check_frames'])].copy()
    roi_root=roi_root or ROOT/config['state_root']/'ROIs'
    with compatibility():
        try:
            result=evaluator.eval(ground,predicted,dstype='validation',roidir=str(roi_root)).iloc[0]
            return {key:float(result[key]) for key in ['idf1','idp','idr','idtp','idfp','idfn','precision','recall','num_switches','num_frames']}
        except ValueError as error:
            if str(error)!='No objects to concatenate':raise
            # Official code cannot concatenate zero surviving prediction cameras.
            # All GT identities are missed in this empty-after-filtering corner case.
            return dict(idf1=0.,idp=0.,idr=0.,idtp=0.,idfp=0.,idfn=float(len(ground)),
                precision=0.,recall=0.,num_switches=0.,num_frames=float(ground[['CameraId','FrameId']].drop_duplicates().shape[0]),
                empty_prediction_after_official_filters=True)


def audit_claims(config,events):
    root=Path(config['dataset_root']);gt={};owners={};counts={key:0 for key in ['correct','false','unscored','enrollment_unscored']}
    for scene in config['scenes']:
        for cam in (root/'validation'/scene).glob('c*'):
            rows=np.loadtxt(cam/'gt/gt.txt',delimiter=',',ndmin=2)
            for row in rows:gt.setdefault((cam.name,int(row[0])),[]).append((int(row[1]),row[2:6]))
    for event in events:
        found=gt.get((event['camera'],event['frame']),[]);actual=None;best=.5
        x,y,w,h=event['box'];box=np.array([x,y,x+w,y+h])
        for identity,raw in found:
            a,b,c,d=raw;other=np.array([a,b,a+c,b+d]);area=np.maximum(0,np.minimum(box[2:],other[2:])-np.maximum(box[:2],other[:2])).prod()
            iou=area/max(1.,w*h+c*d-area)
            if iou>=best:actual=identity;best=iou
        key=(event['scene'],event['identity'])
        if event['action']=='new':
            if actual is None:counts['enrollment_unscored']+=1
            else:owners.setdefault(key,actual)
        else:
            owner=owners.get(key)
            label='unscored' if owner is None or actual is None else 'correct' if owner==actual else 'false'
            counts[label]+=1
    return dict(counts=counts,warning='Sparse accepted-claim audit, IoU>=.5 against labeled MTMC vehicles; unannotated/ambiguous detections remain unscored. Not a substitute for official IDF1.')
