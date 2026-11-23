"""Causal, quality-gated temporal evidence for experimental global recovery."""
from dataclasses import dataclass
import numpy as np
from .association import unit
from .tracklets import TrackletAssociation


@dataclass
class Sample:
    frame:int
    feature:np.ndarray
    weight:float


class TrackletEvidenceAssociation(TrackletAssociation):
    def __init__(self,settings):
        super().__init__(settings)
        self.evidence=settings['aggregation'].copy()
        for key in ['min_samples','max_samples','sample_gap','min_span']:
            if not isinstance(self.evidence[key],int) or self.evidence[key]<1:
                raise ValueError(f'aggregation.{key} must be a positive integer')
        if self.evidence['max_samples']<self.evidence['min_samples']:
            raise ValueError('Aggregation window is shorter than required evidence')
        if not -1<=self.evidence['coherence']<=1:
            raise ValueError('Invalid aggregation coherence threshold')
        self.histories={}
        self.last_good={}
        self.queries={}
        self.view_epoch=0

    def reset_view(self):
        super().reset_view()
        self.histories.clear();self.last_good.clear();self.queries.clear()
        self.view_epoch+=1

    def _pool(self,samples):
        if len(samples)<self.evidence['min_samples'] or samples[-1].frame-samples[0].frame<self.evidence['min_span']:
            return None
        vectors=np.stack([sample.feature for sample in samples])
        weights=np.array([sample.weight for sample in samples],dtype=np.float32)
        mean=(vectors*weights[:,None]).sum(0)/weights.sum()
        if np.linalg.norm(mean)<1e-8:
            return None
        pooled=unit(mean)
        if float((vectors@pooled).min())<self.evidence['coherence']:
            return None
        return pooled

    def _observe_tracklets(self,local_ids,boxes,features,reliable,frame):
        live={item.identity for item in self.local.tracks}
        self.histories={key:value for key,value in self.histories.items() if key in live}
        self.last_good={key:value for key,value in self.last_good.items() if key in live}
        self.queries={}
        for local,box,feature,good in zip(local_ids,boxes,features,reliable):
            if not good:
                self.histories.pop(local,None);self.last_good.pop(local,None)
                continue
            if self.last_good.get(local)!=frame-1:
                self.histories[local]=[]
            self.last_good[local]=frame
            samples=self.histories.setdefault(local,[])
            if not samples or frame-samples[-1].frame>=self.evidence['sample_gap']:
                # More original crop pixels get more weight; upsampling itself
                # never counts as additional visual evidence.
                weight=float(np.sqrt(np.prod(box[2:]-box[:2])))
                samples.append(Sample(frame,feature.copy(),weight))
                self.histories[local]=samples=samples[-self.evidence['max_samples']:]
            pooled=self._pool(samples)
            if pooled is not None:
                self.queries[local]=pooled

    def _freeze_source(self,identity,local):
        item=self.identities[identity]
        if getattr(item,'enrollment_key',None)!=(self.view_epoch,local):
            return
        if getattr(item,'enrollment_pool',None) is not None:
            return
        pooled=self.queries.get(local)
        if pooled is None:
            return
        # Keep original reference vectors immutable alongside frozen consensus.
        item.enrollment_pool=pooled.copy()
        item.enrollment_members=[sample.feature.copy() for sample in self.histories[local]]
        item.enrollment_frames=[sample.frame for sample in self.histories[local]]

    def _enroll(self,local,box,feature,frame):
        identity=super()._enroll(local,box,feature,frame)
        self.identities[identity].enrollment_key=(self.view_epoch,local)
        self._freeze_source(identity,local)
        return identity

    def _touch(self,identity,box,feature,reliable,frame,recovery=False):
        item=self.identities[identity]
        if recovery and self.evidence['quarantine_recovered_memory']:
            item.recovery_write_quarantine=True
        result=super()._touch(identity,box,feature,reliable,frame,recovery)
        if not recovery and reliable:
            local=next((key for key,value in self.bindings.items() if value==identity),None)
            if local is not None:self._freeze_source(identity,local)
        return result

    def _memory_write_allowed(self,item,frame):
        return not getattr(item,'recovery_write_quarantine',False)

    def _recovery_ready(self,item,local,frame):
        return (local in self.queries and self.last_good.get(local)==frame and
                (not self.evidence['pool_source'] or getattr(item,'enrollment_pool',None) is not None))

    def _recovery_similarity(self,item,feature,local,frame):
        pooled=self.queries.get(local)
        if pooled is None:
            # Raw appearance can prevent premature new-ID enrollment while
            # temporal evidence is accumulating; it cannot authorize recovery.
            return item.similarity(feature)
        if not self.evidence['pool_source']:
            return item.similarity(pooled)
        consensus=getattr(item,'enrollment_pool',None)
        if consensus is None:
            return item.similarity(pooled)
        anchor=max(float(value@pooled) for value in item.references)
        source=float(consensus@pooled)
        recent=max([float(value@pooled) for value in item.recent] or [source])
        return .7*source+.3*recent,min(anchor,source)

    def update(self,boxes,features,reliable,frame):
        decisions=super().update(boxes,features,reliable,frame)
        for decision in decisions:
            local=decision['tracklet_id'];samples=self.histories.get(local,[])
            decision['evidence_samples']=len(samples)
            decision['evidence_span']=samples[-1].frame-samples[0].frame if samples else 0
            decision['pooled_query_ready']=local in self.queries
            item=self.identities.get(decision['identity'])
            decision['source_consensus_ready']=item is not None and getattr(item,'enrollment_pool',None) is not None
        return decisions
