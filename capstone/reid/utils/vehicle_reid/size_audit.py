"""Evaluation-only size strata; object size never enters tracker decisions."""
from collections import Counter
from pycocotools import mask as rle


class SizeAudit:
    def __init__(self,threshold):self.threshold=threshold;self.counts={name:Counter() for name in ['small','larger']}
    def update(self,boxes,decisions,prepared,prior_truth):
        for position,(box,decision) in enumerate(zip(boxes,decisions)):
            group='small' if min(box[2]-box[0],box[3]-box[1])<self.threshold else 'larger'
            counts=self.counts[group];actual=prepared['matched'].get(position)
            if actual is not None:
                counts['gt_matched_observations']+=1
                if decision.get('global_identity',decision['identity']) is None:counts['unresolved_global_observations']+=1
                prior=prior_truth.get(decision['identity'])
                if prior is not None and prior!=actual:counts['wrong_identity_observations']+=1
            if decision.get('association_kind')=='global_recovery':
                previous=prior_truth.get(decision['identity'])
                counts['global_accepted']+=1
                label='unscored' if actual is None or previous is None else 'correct' if actual==previous else 'false'
                counts['global_'+label]+=1
    def summary(self):return {name:dict(value) for name,value in self.counts.items()}


def return_sizes(metric,threshold):
    """Classify GT visibility returns by GT box size at the return frame."""
    last={};groups={name:Counter() for name in ['small','larger']};event_groups={}
    for frame,entries in sorted(metric.truth.items()):
        for identity,encoded in entries:
            if identity in last and frame>last[identity]+1:
                box=rle.toBbox(encoded)
                group='small' if min(box[2:])<threshold else 'larger'
                groups[group]['visibility_returns']+=1;event_groups[(identity,frame)]=group
            last[identity]=frame
    for event in metric.events:
        group=event_groups[(event['truth_id'],event['returned_at'])]
        groups[group]['eligible_returns']+=1;groups[group][event['result']]+=1
    for identity,event in metric.pending.items():
        group=event_groups[(identity,event['returned_at'])]
        groups[group]['eligible_returns']+=1;groups[group]['censored']+=1
    result={}
    for name,counts in groups.items():
        item=dict(counts);item['missed_returns']=counts['visibility_returns']-counts['recovered']-counts['censored']
        result[name]=item
    return result
