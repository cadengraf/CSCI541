from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import yaml
from utils.tracking.common import ROOT,load
from utils.experiments.vehicle_recovery.command import replay
from utils.experiments.vehicle_recovery.metrics import Metrics


def decision(identity,kind='local_continuation',status='matched'):
    return dict(identity=identity,association_kind=kind,status=status,memory_update=False,similarity=.9,margin=.2)


class RecoveryTests(unittest.TestCase):
    def setUp(self):self.spec=dict(minimum_gap_frames=2,long_gap_frames=16,recovery_window_frames=3)

    def test_delayed_natural_return_uses_original_identity(self):
        m=Metrics(self.spec,1,'natural',[10,20]);o=[{'truth':1}]
        m.update(1,[1],o,[decision(10)],[True]);m.update(2,[1],o,[decision(11)],[True])
        for f in [3,4]:m.update(f,[],[],[],[])
        m.update(5,[1],o,[decision(11)],[True]);m.update(6,[1],o,[decision(10,'global_recovery','recovered')],[True])
        r=m.summary(6);e=r['events'][0]
        self.assertEqual((e['expected'],e['result'],e['delay_frames']),(10,'recovered',1))
        self.assertEqual(r['counts']['global_correct'],1)

    def test_blackout_is_not_a_natural_gap_and_new_object_false_claim_is_counted(self):
        m=Metrics(self.spec,1,'target_blackout',[2,5]);o=[{'truth':1}]
        m.update(1,[1],o,[decision(10)],[True])
        for f in [2,3,4]:m.update(f,[1],[],[],[],hidden=True)
        m.update(5,[1],o,[decision(None,status='rejected')],[True])
        m.update(6,[1,2],[{'truth':2}],[decision(10,'global_recovery','recovered')],[True])
        m.update(7,[1],o,[decision(None,status='rejected')],[True])
        r=m.summary(7)
        self.assertEqual(r['returns']['natural_annotation']['events'],0)
        self.assertEqual(r['returns']['controlled_blackout']['missed'],1)
        self.assertEqual(r['counts']['global_false'],1)
        self.assertEqual(r['counts']['false_reconnection_to_new_object'],1)
        self.assertEqual(r['counts']['target_false_match_frames'],1)

    def test_unenrolled_and_censored_returns_are_separate(self):
        m=Metrics(self.spec,1,'natural',[10,20])
        m.update(1,[1,2],[{'truth':1}],[decision(10)],[True]);m.update(2,[],[],[],[]);m.update(3,[],[],[],[])
        m.update(4,[1,2],[],[],[]);r=m.summary(4)
        self.assertEqual(r['returns']['natural_annotation']['unenrolled'],1)
        self.assertEqual(r['returns']['natural_annotation']['censored'],1)

    def test_real_association_blackout_and_exact_checkpoint_resume(self):
        config=yaml.safe_load((ROOT/'configs/vehicle_recovery.yaml').read_text())
        config=deepcopy(config);config['evaluation']['recovery_window_frames']=10
        frames=[];bank=[]
        for frame in range(1,71):
            observations=[dict(truth=1,box=[10,10,70,70])]
            frames.append(dict(frame=frame,truth=[1],path='unused',observations=observations,reliable=[True]))
            bank.append(torch.tensor([[1.,0.,0.]]))
        case=dict(target=1,blackout=[10,55],frames=frames);selection=dict(cases={'test':case})
        with tempfile.TemporaryDirectory() as directory:
            config.update(state_root=directory+'/state',output=directory+'/output',checkpoint_frames=5)
            Path(config['output']).mkdir();(Path(config['output'])/'selection.pt').write_bytes(b'frozen toy selection')
            from utils.experiments.vehicle_recovery import command
            original_save=command.save
            def interrupt(payload,path):
                original_save(payload,path)
                if payload['next_frame']==20:raise RuntimeError('interrupted')
            with patch.object(command,'save',interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):replay(config,selection,'test','target_blackout',bank)
            state=replay(config,selection,'test','target_blackout',bank)
            self.assertEqual(state['summary']['reference']['returns']['controlled_blackout']['missed'],1)
            self.assertEqual(state['summary']['query_pool']['returns']['controlled_blackout']['recovered'],1)
            self.assertEqual(state['summary']['query_pool']['events'][0]['delay_frames'],4)
            config['state_root']=directory+'/uninterrupted'
            full=replay(config,selection,'test','target_blackout',bank)
            self.assertEqual(state['rows'],full['rows']);self.assertEqual(state['summary'],full['summary'])
            again=replay(config,selection,'test','target_blackout',bank);self.assertEqual(again['rows'],full['rows'])


if __name__=='__main__':unittest.main()
