"""GT-visibility-gap recovery metrics, isolated from inference and MOT scoring."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as rle


class Recovery:
    def __init__(self, truth_path, window=10):
        self.truth = {}
        for line in truth_path.read_text().splitlines():
            frame, identity, category, height, width, counts = line.split()
            if int(category) == 1:
                self.truth.setdefault(int(frame), []).append((int(identity),
                    dict(size=[int(height), int(width)], counts=counts.encode())))
        self.window = window
        self.counts = Counter()
        self.last_seen = {}
        self.last_identity = {}
        self.identity_truth = {}
        self.identity_last = {}
        self.pending = {}
        self.events = []
        self.action_counts = Counter()
        self.global_identity_groups = {}
        self.audit_frames = 0

    @staticmethod
    def bucket(gap):
        return '1-5' if gap <= 5 else '6-15' if gap <= 15 else '16-60' if gap <= 60 else '61+'

    def finish_event(self, identity, result, frame):
        event = self.pending.pop(identity)
        event.update(result=result, decided_at=frame)
        self.events.append(event)
        self.counts[result] += 1
        self.counts[f'{event["bucket"]}_{result}'] += 1

    def prepare_frame(self, frame, masks):
        """Evaluation-only IoU labels reusable across methods emitting the same masks."""
        truth = [(identity, rle.decode(encoded).astype(bool)) for identity, encoded in self.truth.get(frame, [])]
        matched = {}
        if len(masks) and truth:
            overlaps = np.zeros((len(masks),len(truth)))
            for row,mask in enumerate(masks):
                for column,(_,target) in enumerate(truth):
                    overlaps[row,column] = np.logical_and(mask,target).sum()/max(1,np.logical_or(mask,target).sum())
            rows,columns = linear_sum_assignment(1-overlaps)
            matched = {int(row):truth[column][0] for row,column in zip(rows,columns) if overlaps[row,column]>=.5}
        return dict(frame=frame,count=len(masks),truth_ids=[identity for identity,_ in truth],matched=matched)

    def audit_actions(self, frame, matched, decisions):
        if not hasattr(self,'action_counts'):
            self.action_counts,self.global_identity_groups,self.audit_frames = Counter(),{},0
        self.audit_frames += 1
        for row,decision in enumerate(decisions):
            kind = decision.get('association_kind')
            if decision.get('status') == 'pending_recovery':
                self.action_counts['pending_global_observations'] += 1
            if decision.get('status') in ('ambiguous','rejected'):
                self.action_counts['rejected_global_observations'] += 1
            if kind is None and decision.get('status') == 'recovered':
                kind = 'global_recovery'
            if kind not in ('global_recovery','local_reacquisition'):
                continue
            prefix = 'global' if kind == 'global_recovery' else 'local'
            self.action_counts[f'{prefix}_accepted'] += 1
            identity = decision.get('identity')
            actual = matched.get(row)
            previous = self.identity_truth.get(identity)
            if actual is None or previous is None:
                self.action_counts[f'{prefix}_unscored'] += 1
                continue
            wrong = previous != actual
            self.action_counts[f'{prefix}_false' if wrong else f'{prefix}_correct'] += 1
            if prefix == 'global':
                group = self.global_identity_groups.setdefault(actual,dict(accepted=0,false=0))
                group['accepted'] += 1
                group['false'] += int(wrong)

    def action_summary(self):
        if not getattr(self,'audit_frames',0):
            return dict(available=False,note='Legacy checkpoint has no separate local/global action audit')
        result = dict(self.action_counts)
        correct,false = self.action_counts['global_correct'],self.action_counts['global_false']
        scored = correct+false
        groups = list(self.global_identity_groups.values())
        false_groups = sum(group['false']>0 for group in groups)
        from scipy.stats import beta
        upper = (1. if not groups or false_groups==len(groups) else
                 float(beta.ppf(.95,false_groups+1,len(groups)-false_groups)))
        result.update(available=True,frames=self.audit_frames,global_scored=scored,
            global_correct=correct,global_false=false,global_unscored=self.action_counts['global_unscored'],
            global_accepted=self.action_counts['global_accepted'],local_correct=self.action_counts['local_correct'],
            local_false=self.action_counts['local_false'],local_unscored=self.action_counts['local_unscored'],
            global_false_rate=false/scored if scored else None,
            global_scored_identities=len(groups),global_false_identities=false_groups,
            identity_any_false_upper95=upper,
            uncertainty_note='Identity-level any-error diagnostic; repeated recovery events are correlated; not a distributional guarantee')
        return result

    def update(self, frame, masks, decisions, prepared=None):
        if len(masks) != len(decisions):
            raise ValueError('Masks and decisions have different lengths')
        prepared = self.prepare_frame(frame,masks) if prepared is None else prepared
        if prepared['frame'] != frame or prepared['count'] != len(masks):
            raise ValueError('Prepared evaluation labels use another frame or mask count')
        truth = [(identity,None) for identity in prepared['truth_ids']]
        actual_ids = set(prepared['truth_ids'])
        for identity in list(self.pending):
            if identity not in actual_ids:
                self.finish_event(identity,'failed',frame)
        matched = prepared['matched']
        self.audit_actions(frame,matched,decisions)
        by_truth = {actual: decisions[row]['identity'] for row, actual in matched.items()}
        for identity, _ in truth:
            previous = self.last_seen.get(identity)
            if previous is not None and frame > previous+1:
                gap = frame-previous-1
                self.counts['visibility_returns'] += 1
                expected = self.last_identity.get(identity)
                if expected is None:
                    self.counts['returns_without_prior_match'] += 1
                else:
                    self.pending[identity] = dict(truth_id=identity, returned_at=frame, gap=gap,
                        bucket=self.bucket(gap), expected=expected, visible_steps=0)
                    self.counts['eligible_returns'] += 1
                    self.counts[f'{self.bucket(gap)}_eligible'] += 1
            if identity in self.pending:
                event = self.pending[identity]
                event['visible_steps'] += 1
                if by_truth.get(identity) == event['expected']:
                    event['delay_frames'] = frame-event['returned_at']
                    self.finish_event(identity, 'recovered', frame)
                elif event['visible_steps'] >= self.window:
                    self.finish_event(identity, 'failed', frame)
            self.last_seen[identity] = frame
        for row, actual in matched.items():
            identity = decisions[row]['identity']
            if decisions[row].get('global_identity', identity) is None:
                self.counts['matched_unknown_observations'] += 1
            if identity is None:
                continue
            prior_truth = self.identity_truth.get(identity)
            prior_frame = self.identity_last.get(identity)
            if prior_frame is not None and frame > prior_frame+1:
                self.counts['correct_reconnections' if prior_truth == actual else 'false_reconnections'] += 1
            if prior_truth is not None and prior_truth != actual:
                self.counts['wrong_identity_observations'] += 1
            self.identity_truth.setdefault(identity, actual)
            self.identity_last[identity] = frame
            self.last_identity[actual] = identity

    def summary(self):
        result = dict(self.counts)
        eligible = self.counts['eligible_returns']
        recovered = self.counts['recovered']
        reconnects = self.counts['correct_reconnections'] + self.counts['false_reconnections']
        result.update(eligible_returns=eligible, recovered=recovered, failed=self.counts['failed'],
            censored=len(self.pending), recovery_rate=recovered/eligible if eligible else None,
            reconnection_precision=self.counts['correct_reconnections']/reconnects if reconnects else None,
            mean_recovery_delay=float(np.mean([event['delay_frames'] for event in self.events if event['result']=='recovered'])) if recovered else None)
        return result
