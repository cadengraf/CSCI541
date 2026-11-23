from copy import deepcopy
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
import yaml
import torch
from utils.tracking.common import ROOT,save
from utils.experiments.cityflow.association import MultiCamera
from utils.experiments.cityflow.data import schedule,capture,detections
from utils.experiments.cityflow.evaluation import score


class CityFlowTests(unittest.TestCase):
    def setUp(self):self.config=yaml.safe_load((ROOT/'configs/cityflow.yaml').read_text())

    def test_chronological_scheduler_resumes_exact_suffix(self):
        cameras=dict(c001=dict(offset=0.,fps=10.,frames=3),c002=dict(offset=.05,fps=10.,frames=2))
        full=list(schedule(cameras,dict(c001=1,c002=1)))
        self.assertEqual([(c,f) for _,c,f in full],[('c001',1),('c002',1),('c001',2),('c002',2),('c001',3)])
        self.assertEqual(full[2:],list(schedule(cameras,dict(c001=2,c002=2))))

    def actor(self):return MultiCamera(self.config,dict(threshold=.8,margin=.08),['c001','c002'])

    def step(self,actor,frame,cam='c001',good=True,feature=None):
        box=np.array([[10,10,70,70] if cam=='c001' else [1000,1000,1060,1060]],np.float32)
        feature=np.array([[1.,0.,0.]] if feature is None else [feature],np.float32)
        return actor.update(cam,frame,float(frame),box,feature,[good])[0]

    def test_overlapping_views_share_identity_without_cross_camera_geometry(self):
        actor=self.actor()
        for f in range(1,6):
            a=self.step(actor,f,'c001');b=self.step(actor,f,'c002')
        self.assertEqual((a['global_identity'],b['global_identity']),(1,1))
        self.assertEqual(b['action'],'cross_camera');self.assertFalse(b['memory_update'])
        self.assertEqual(actor.profiles[1]['cameras'],{'c001','c002'})

    def test_identical_competitors_are_rejected_and_same_camera_ids_are_unique(self):
        actor=self.actor();boxes=np.array([[10,10,70,70],[200,10,260,70]],np.float32);vectors=np.array([[1.,0.,0.],[1.,0.,0.]],np.float32)
        for f in range(1,6):rows=actor.update('c001',f,float(f),boxes,vectors,[True,True])
        self.assertEqual(len({r['global_identity'] for r in rows}),2)
        for f in range(6,11):row=self.step(actor,f,'c002')
        self.assertIsNone(row['global_identity']);self.assertEqual(row['action'],'no_confident_match')

    def test_unreliable_observation_preserves_appearance_and_blocks_memory_write(self):
        actor=self.actor()
        for f in range(1,6):self.step(actor,f)
        reference=actor.profiles[1]['reference'].copy();old=actor.local['c001'].tracks[0].feature.clone()
        row=self.step(actor,6,good=False,feature=[0.,1.,0.])
        np.testing.assert_array_equal(reference,actor.profiles[1]['reference'])
        np.testing.assert_array_equal(old.numpy(),actor.local['c001'].tracks[0].feature.numpy());self.assertFalse(row['memory_update'])

    def test_pending_evidence_and_overlap_ownership_survive_pickle_resume(self):
        actor=self.actor()
        for f in range(1,5):self.step(actor,f,'c001');self.step(actor,f,'c002')
        restored=pickle.loads(pickle.dumps(actor))
        for f in range(5,9):
            for cam in ['c001','c002']:self.assertEqual(self.step(actor,f,cam),self.step(restored,f,cam))
        self.assertEqual(actor.counts,restored.counts)
        with self.assertRaisesRegex(ValueError,'timestamp order'):self.step(actor,1)

    def test_official_evaluator_perfect_and_single_camera_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);roi=root/'ROIs'
            for camera in ['c001','c002']:
                folder=roi/'validation'/camera;folder.mkdir(parents=True);Image.new('L',(100,100),255).save(folder/'roi.jpg')
            gt=root/'gt.txt';pred=root/'pred.txt'
            gt.write_text(''.join(f'{cam} 1 {frame} 10 10 20 20 -1 -1\n' for cam in [1,2] for frame in [1,2]))
            pred.write_text(''.join(f'{cam} 99 {frame} 10 10 20 20 -1 -1\n' for cam in [1,2] for frame in [1,2]))
            result=score(self.config,pred,gt,roi)
            self.assertAlmostEqual(result['idf1'],1.);self.assertAlmostEqual(result['idp'],1.);self.assertAlmostEqual(result['idr'],1.)
            pred.write_text('1 99 1 10 10 20 20 -1 -1\n')
            result=score(self.config,pred,gt,roi);self.assertEqual(result['idf1'],0.)
            self.assertTrue(result['empty_prediction_after_official_filters'])

    def test_real_video_seek_reproduces_sequential_frame(self):
        path=Path(self.config['dataset_root'])/'validation/S02/c006/vdo.avi';cap=capture(path)
        try:
            for _ in range(51):okay,expected=cap.read()
        finally:cap.release()
        cap=capture(path,51)
        try:okay,actual=cap.read()
        finally:cap.release()
        self.assertTrue(okay);np.testing.assert_array_equal(actual,expected)

    def test_prediction_transactions_resume_exactly_on_real_video_inputs(self):
        from utils.experiments.cityflow import command
        class Encoder(torch.nn.Module):
            def __init__(self):super().__init__();self.marker=torch.nn.Parameter(torch.ones(1))
            def forward(self,batch):return torch.stack([self.marker[0]*torch.ones(len(batch)),torch.zeros(len(batch)),torch.zeros(len(batch))],dim=1)
        def model(*args):return Encoder(),lambda value:torch.zeros(3,8,8)
        root=Path(self.config['dataset_root']);cameras={}
        for cam in ['c006','c007']:
            folder=root/'validation/S02'/cam
            cameras[cam]=dict(path=str(folder/'vdo.avi'),frames=20,fps=10.,offset=0.,size=[1920,1080],
                detections=str(folder/'det/det_yolo3.txt'),roi=str(folder/'roi.jpg'))
        selection=dict(models={'toy':dict(kind='test',dimension=3,point=dict(threshold=.8,margin=.1),sha256='test',training='unit test only')},scenes={'S02':cameras})
        with tempfile.TemporaryDirectory() as directory:
            config=deepcopy(self.config);config.update(output=directory+'/out',state_root=directory+'/resumed',checkpoint_frames=10)
            Path(config['output']).mkdir();save(selection,Path(config['output'])/'selection.pt')
            original=MultiCamera.update;calls=0
            def interrupt(actor,*args):
                nonlocal calls
                calls+=1
                if calls==13:raise RuntimeError('interrupted after committed prediction chunk')
                return original(actor,*args)
            with patch.object(command,'build_model',model),patch('utils.experiments.cityflow.evaluation.score',return_value={}),patch('utils.experiments.cityflow.evaluation.audit_claims',return_value={}):
                with patch.object(MultiCamera,'update',interrupt):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):command.track(config,selection,'toy')
                command.track(config,selection,'toy');pred=Path(config['output'])/'toy/predictions.txt';resumed=pred.read_bytes()
                config['state_root']=directory+'/full';command.track(config,selection,'toy');self.assertEqual(resumed,pred.read_bytes())
                with patch.object(command,'build_model',side_effect=AssertionError('completed stages must not reload encoder')):
                    command.track(config,selection,'toy');self.assertEqual(resumed,pred.read_bytes())


if __name__=='__main__':unittest.main()
