"""Online diagnostics with ground truth isolated from the identity algorithm."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment


class Diagnostics:
    def __init__(self, spec):
        self.spec = spec
        self.counts = Counter()
        self.identity_truth = {}
        self.truth_last = {}
        self.truth_identity = {}
        self.pending_returns = {}
        self.truth_original = {}
        self.rows = None
        if spec['truth']:
            from utils.tracking.common import ROOT
            self.rows = {}
            for line in (ROOT / spec['truth']).read_text().splitlines():
                frame, identity, category, height, width, counts = line.split()
                if int(category) == 1:
                    self.rows.setdefault(int(frame), []).append((int(identity), dict(
                        size=[int(height), int(width)], counts=counts.encode('ascii'))))

    def update(self, frame, masks, decisions, target):
        self.counts.update(decision['status'] for decision in decisions)
        if self.rows is None:
            return ['unlabeled'] * len(decisions)
        from pycocotools import mask as rle
        truth = [(identity, rle.decode(encoded).astype(bool)) for identity, encoded in self.rows.get(frame, [])]
        matches = {}
        if len(masks) and truth:
            overlaps = np.zeros((len(masks), len(truth)))
            for row, mask in enumerate(masks):
                for column, (_, actual) in enumerate(truth):
                    overlaps[row, column] = np.logical_and(mask, actual).sum()/max(1, np.logical_or(mask, actual).sum())
            rows, columns = linear_sum_assignment(1-overlaps)
            matches = {row: truth[column][0] for row, column in zip(rows, columns)
                       if overlaps[row, column] >= self.spec['min_mask_iou']}
        labels = []
        original_at_start = dict(self.truth_original)
        by_truth = {identity: decisions[row]['identity'] for row, identity in matches.items()}
        for row, decision in enumerate(decisions):
            identity, actual = decision['identity'], matches.get(row)
            label = 'unknown' if identity is None else 'unlabeled_detection'
            if identity is not None and actual is not None:
                previous = self.identity_truth.get(identity)
                label = 'correct' if previous is None or previous == actual else 'wrong_identity'
                self.counts[label] += 1
                if decision['status'] == 'recovered':
                    self.counts['correct_reconnection' if previous == actual else
                                'false_reconnection' if previous is not None else 'unscored_reconnection'] += 1
                self.identity_truth.setdefault(identity, actual)
                previous_identity = self.truth_identity.get(actual)
                if previous_identity is not None and identity != previous_identity:
                    self.counts['identity_switch_observations'] += 1
                self.truth_identity[actual] = identity
                self.truth_original.setdefault(actual, identity)
            labels.append(label)
        for actual, _ in truth:
            previous = self.truth_last.get(actual)
            returning = previous is not None and frame > previous+1
            if returning:
                self.counts['return_events'] += 1
            if returning:
                original = original_at_start.get(actual)
                if original is not None:
                    self.counts['recovered_on_return' if by_truth.get(actual) == original else 'missed_on_return'] += 1
                    if by_truth.get(actual) != original:
                        self.pending_returns[actual] = (original, frame)
            if actual in self.pending_returns:
                original, returned_at = self.pending_returns[actual]
                if by_truth.get(actual) == original:
                    self.counts['delayed_recoveries'] += 1
                    self.counts['recovery_delay_frames_total'] += frame-returned_at
                    del self.pending_returns[actual]
                else:
                    self.counts['missed_recovery_visible_frames'] += 1
            self.truth_last[actual] = frame
        target_truth = self.spec['target_truth_id']
        if target_truth is not None:
            target_truth = int(target_truth)
            selected = [matches.get(row) for row, decision in enumerate(decisions) if decision['identity'] == target]
            present = any(identity == target_truth for identity, _ in truth)
            if selected:
                self.counts['target_correct' if selected[0] == target_truth else 'target_false_match'] += 1
            elif present:
                self.counts['target_missed_visible'] += 1
        return labels
