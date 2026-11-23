"""Evaluation-only immutable identity ownership and original-ID recovery events."""
from collections import Counter
import numpy as np


class Metrics:
    def __init__(self,spec,target,scenario,blackout):
        self.spec=spec;self.target=target;self.scenario=scenario;self.blackout=blackout
        self.owners={};self.original={};self.last_seen={};self.last_claim={};self.pending={};self.events=[]
        self.counts=Counter();self.target_identity=None

    def finish(self,actual,result,frame):
        event=self.pending.pop(actual);event.update(result=result,decided_at=frame)
        if result=='recovered':event['delay_frames']=frame-event['returned_at']
        if result=='missed':
            event['reason']=('no_crop_observation' if not event['observations'] else
                'no_reliable_crop' if not event['reliable_observations'] else
                'local_fragmentation' if 'reference' in event['statuses'] else
                'insufficient_temporal_evidence' if not event['pooled_ready_observations'] else
                'ambiguous_candidates' if 'ambiguous' in event['statuses'] else
                'different_identity_bound' if event['other_identity_observations'] else 'appearance_or_reference_threshold')
        self.events.append(event)

    def update(self,frame,truth,observations,decisions,reliable,hidden=False):
        if len(observations)!=len(decisions) or len(decisions)!=len(reliable):raise ValueError('Scoring alignment mismatch')
        actual_ids=set(truth)-({self.target} if hidden else set())
        by_truth={o['truth']:(decision,good) for o,decision,good in zip(observations,decisions,reliable)}
        previous_owners=dict(self.owners);seen_before=set(self.last_seen)
        for actual in list(self.pending):
            if actual not in actual_ids:self.finish(actual,'censored',frame)
        for actual in sorted(actual_ids):
            previous=self.last_seen.get(actual)
            if previous is not None and frame-previous-1>=self.spec['minimum_gap_frames']:
                expected=self.original.get(actual);gap=frame-previous-1
                controlled=(self.scenario=='target_blackout' and actual==self.target and previous<self.blackout[1]<=frame and previous<self.blackout[0])
                event=dict(truth=actual,returned_at=frame,gap_frames=gap,expected=expected,
                    kind='controlled_blackout' if controlled else 'natural_annotation',long_gap=gap>=self.spec['long_gap_frames'],
                    observations=0,reliable_observations=0,pooled_ready_observations=0,other_identity_observations=0,
                    statuses=[],max_similarity=None,max_margin=None)
                if expected is None:
                    event.update(result='unenrolled',decided_at=frame);self.events.append(event)
                else:self.pending[actual]=event
            if actual in self.pending:
                event=self.pending[actual];observed=by_truth.get(actual)
                if observed:
                    decision,good=observed;identity=decision['identity'];event['observations']+=1;event['reliable_observations']+=int(good)
                    event['pooled_ready_observations']+=int(decision.get('pooled_query_ready',False))
                    event['other_identity_observations']+=int(identity is not None and identity!=event['expected'])
                    if decision['status'] not in event['statuses']:event['statuses'].append(decision['status'])
                    for name in ['similarity','margin']:
                        value=float(decision.get(name,0.));key='max_'+name
                        event[key]=max(value,event[key]) if event[key] is not None else value
                    if identity==event['expected']:self.finish(actual,'recovered',frame)
                if actual in self.pending and frame-event['returned_at']+1>=self.spec['recovery_window_frames']:
                    self.finish(actual,'missed',frame)
            self.last_seen[actual]=frame
        for observation,decision,good in zip(observations,decisions,reliable):
            actual=observation['truth'];identity=decision['identity'];kind=decision.get('association_kind')
            self.counts['observations']+=1;self.counts['memory_updates']+=int(decision['memory_update'])
            if identity is None:self.counts['no_confident_identity']+=1
            else:
                owner=previous_owners.get(identity)
                if owner is not None and owner!=actual:self.counts['wrong_identity_observations']+=1
                if kind in ['global_recovery','local_reacquisition']:
                    prefix='global' if kind=='global_recovery' else 'local'
                    category='unscored' if owner is None else 'correct' if owner==actual else 'false'
                    self.counts[prefix+'_'+category]+=1
                    if category=='false' and actual not in seen_before:self.counts['false_reconnection_to_new_object']+=1
                self.owners.setdefault(identity,actual)
                if good and self.owners[identity]==actual:self.original.setdefault(actual,identity)
                self.last_claim[actual]=identity
            enrollment_open=self.scenario!='target_blackout' or frame<self.blackout[0]
            if enrollment_open and actual==self.target and self.target_identity is None and good and identity is not None and self.owners.get(identity)==actual:
                self.target_identity=identity
        claimed=[o['truth'] for o,d in zip(observations,decisions) if self.target_identity is not None and d['identity']==self.target_identity]
        if claimed:
            self.counts['target_correct_frames' if claimed==[self.target] else 'target_false_match_frames']+=1
        elif self.target in actual_ids:self.counts['target_missed_visible_frames']+=1
        if hidden:self.counts['target_blackout_frames']+=1

    def summary(self,last_frame):
        for actual in list(self.pending):self.finish(actual,'censored',last_frame)
        def group(events):
            counts=Counter(e['result'] for e in events);eligible=sum(counts[key] for key in ['recovered','missed','censored'])
            delays=[e['delay_frames'] for e in events if e['result']=='recovered']
            return dict(events=len(events),eligible=eligible,recovered=counts['recovered'],missed=counts['missed'],censored=counts['censored'],
                unenrolled=counts['unenrolled'],mean_delay_frames=float(np.mean(delays)) if delays else None,
                missed_reasons=dict(Counter(e['reason'] for e in events if e['result']=='missed')))
        return dict(counts=dict(self.counts),returns={kind:group([e for e in self.events if e['kind']==kind]) for kind in ['natural_annotation','controlled_blackout']},
            long_returns={kind:group([e for e in self.events if e['kind']==kind and e['long_gap']]) for kind in ['natural_annotation','controlled_blackout']},
            target_identity=self.target_identity,events=self.events)
