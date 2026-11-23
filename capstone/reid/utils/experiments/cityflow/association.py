"""Causal multi-camera appearance baseline; camera coordinates never cross views."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from utils.full_frame.tracker import VehicleTracker


def unit(value):
    norm=np.linalg.norm(value)
    if not np.isfinite(value).all() or norm<1e-8:raise ValueError('Invalid identity vector')
    return value/norm


class MultiCamera:
    def __init__(self,config,point,cameras,namespace=0):
        self.config=config;self.point=point;self.local={cam:VehicleTracker(config['local']) for cam in cameras}
        self.bindings={};self.histories={};self.last_good={};self.profiles={};self.next_id=namespace+1;self.counts=Counter();self.last_timestamp=-float('inf')

    def pool(self,key):
        values=self.histories.get(key,[]);spec=self.config['memory']
        if len(values)<spec['min_samples'] or values[-1][0]-values[0][0]<spec['min_span']:return None
        vectors=np.stack([item[1] for item in values]);value=unit(vectors.mean(0))
        return value if float((vectors@value).min())>=spec['coherence'] else None

    def update(self,camera,frame,timestamp,boxes,features,reliable):
        if timestamp<self.last_timestamp-1e-8:raise ValueError('Cameras must advance in global timestamp order')
        self.last_timestamp=timestamp;tracker=self.local[camera];spec=self.config['memory']
        previous={track.identity:track.feature.clone() for track in tracker.tracks}
        local_ids=tracker.update(dict(boxes=torch.as_tensor(boxes),embeddings=torch.as_tensor(features)),frame)
        active={track.identity for track in tracker.tracks}
        for key in list(self.bindings):
            if key[0]==camera and key[1] not in active:self.bindings.pop(key,None)
        for key in list(self.histories):
            if key[0]==camera and key[1] not in active:self.histories.pop(key,None);self.last_good.pop(key,None)
        unreliable={identity for identity,good in zip(local_ids,reliable) if not good}
        for track in tracker.tracks:
            if track.identity in unreliable and track.identity in previous:track.feature=previous[track.identity]
        for identity,feature,good in zip(local_ids,features,reliable):
            key=(camera,identity)
            if not good:self.histories.pop(key,None);self.last_good.pop(key,None);continue
            if self.last_good.get(key)!=frame-1:self.histories[key]=[]
            self.last_good[key]=frame;samples=self.histories.setdefault(key,[])
            if not samples or frame-samples[-1][0]>=spec['sample_gap']:samples.append((frame,unit(feature).copy()))
            self.histories[key]=samples[-spec['max_samples']:]
        keys=[(camera,identity) for identity in local_ids];ready={key:self.pool(key) for key in keys}
        occupied={self.bindings[key] for key in keys if key in self.bindings};unresolved=[i for i,key in enumerate(keys) if key not in self.bindings]
        candidates=[identity for identity in self.profiles if identity not in occupied]
        scores=np.full((len(candidates),len(unresolved)),-1.,dtype=np.float32);anchors=scores.copy()
        allowed=np.zeros(scores.shape,dtype=bool)
        for row,identity in enumerate(candidates):
            profile=self.profiles[identity]
            for col,index in enumerate(unresolved):
                query=ready[keys[index]]
                if query is None:continue
                anchor=float(profile['reference']@query);recent=max([float(value@query) for value in profile['recent']] or [anchor])
                score=.7*anchor+.3*recent;scores[row,col]=score;anchors[row,col]=anchor
                allowed[row,col]=score>=self.point['threshold'] and anchor>=self.point['threshold']
        actions={}
        if len(candidates) and len(unresolved):
            rows,columns=linear_sum_assignment(np.where(allowed,1-scores,1e6))
            for row,col in zip(rows,columns):
                if not allowed[row,col]:continue
                others=np.r_[np.delete(scores[:,col],row),np.delete(scores[row],col)]
                margin=float(scores[row,col]-(others.max() if len(others) else -1.))
                if margin<self.point['margin']:self.counts['ambiguous']+=1;continue
                index=unresolved[col];key=keys[index];identity=candidates[row]
                # At most one local owner in this camera; simultaneous other-camera views are valid.
                self.bindings={k:v for k,v in self.bindings.items() if not (k[0]==camera and v==identity)}
                self.bindings[key]=identity;profile=self.profiles[identity]
                kind='cross_camera' if camera not in profile['cameras'] else 'same_camera_recovery'
                actions[index]=kind;profile['cameras'].add(camera);self.counts[kind]+=1
        for col,index in enumerate(unresolved):
            key=keys[index]
            if key in self.bindings:continue
            query=ready[key]
            if query is None:self.counts['waiting_evidence']+=1;continue
            maximum=float(scores[:,col].max()) if len(candidates) else -1.
            if maximum>=spec['new_identity_max']:self.counts['no_confident_match']+=1;continue
            identity=self.next_id;self.next_id+=1;self.bindings[key]=identity
            self.profiles[identity]=dict(reference=query.copy(),recent=[],cameras={camera})
            actions[index]='new';self.counts['new_profiles']+=1
        result=[]
        for index,key in enumerate(keys):
            identity=self.bindings.get(key);kind=actions.get(index,'continuation' if identity is not None else 'no_confident_match')
            write=False;query=ready[key]
            if identity is not None and query is not None and reliable[index] and index not in actions:
                profile=self.profiles[identity];anchor=float(profile['reference']@query)
                recent=max([float(value@query) for value in profile['recent']] or [anchor]);score=.7*anchor+.3*recent
                competitor=max([float(p['reference']@query) for i,p in self.profiles.items() if i!=identity] or [-1.])
                if score>=spec['write_similarity'] and anchor>=self.point['threshold'] and score-competitor>=spec['write_margin']:
                    profile['recent'].append(query.copy());profile['recent']=profile['recent'][-spec['recent_size']:];write=True;self.counts['memory_updates']+=1
            output=identity if identity is not None else 1_000_000_000+int(camera[1:])*1_000_000+key[1]
            result.append(dict(identity=int(output),global_identity=identity,local_identity=key[1],action=kind,memory_update=write))
        if len({row['identity'] for row in result})!=len(result):raise ValueError('Duplicate same-camera output identity')
        return result
