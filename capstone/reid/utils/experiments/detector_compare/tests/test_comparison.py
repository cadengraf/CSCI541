from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import torch
from pycocotools import mask as rle
import yaml
from utils.experiments.detector_compare import command
from utils.experiments.detector_compare.models import rfdetr_predictions,identity_predictions
from utils.experiments.detector_compare.evaluation import sequence,video_frame


class ComparisonTests(unittest.TestCase):
    def raw(self):
        masks=torch.zeros(1,1,10,10,dtype=torch.bool);masks[:,:,2:8,2:8]=True
        return dict(boxes=torch.tensor([[2.,2.,8.,8.]]),scores=torch.tensor([.9]),masks=masks)

    def encoder(self,image,boxes):return np.ones((len(boxes),4),dtype=np.float32)

    def test_sparse_coco_car_id_and_original_size_masks(self):
        masks=np.ones((3,10,10),dtype=bool)
        result=SimpleNamespace(class_id=np.array([2,3,3]),confidence=np.array([.9,.2,.8]),
                               xyxy=np.array([[0,0,9,9]]*3),mask=masks)
        pred=rfdetr_predictions(result,(10,10,3),3,.3)
        self.assertEqual(pred['boxes'].shape,(1,4))
        self.assertEqual(pred['masks'].shape,(1,1,10,10))
        self.assertAlmostEqual(pred['scores'][0].item(),.8)
        result.mask=np.zeros((3,5,5),dtype=bool)
        with self.assertRaises(ValueError):rfdetr_predictions(result,(10,10,3),3,.3)

    def test_video_padding_preserves_all_original_pixels(self):
        image=np.full((375,1241,3),123,dtype=np.uint8)
        padded=video_frame(image)
        self.assertEqual(padded.shape,(376,1242,3))
        np.testing.assert_array_equal(padded[:375,:1241],image)

    def test_manifest_roundtrip_resumes_and_rejects_changed_gt(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);frames=root/'dataset/training/image_02/0002';frames.mkdir(parents=True)
            cv2.imwrite(str(frames/'000000.png'),np.zeros((10,10,3),dtype=np.uint8))
            labels=root/'dataset/instances_txt';labels.mkdir();truth=labels/'0002.txt';truth.write_text('fixture')
            weights=root/'weights.pt';weights.write_bytes(b'fixture')
            selection=root/'selection.yaml';selection.write_text(yaml.safe_dump(dict(candidate=dict(
                checkpoint=str(weights),checkpoint_sha256=command.digest(weights)))))
            tracking=root/'tracker.yaml';tracking.write_text('association: {local: {max_age: 15}}')
            config=yaml.safe_load((command.ROOT/'configs/detector_compare.yaml').read_text())
            config.update(dataset_root=str(root/'dataset'),output=str(root/'output'),sequences=[2],
                          specialist_selection=str(selection),tracking_config=str(tracking))
            for name in ['codino','rfdetr']:config[name]['checkpoint']=str(weights)
            with patch.object(command.subprocess,'check_output',return_value='{"python":"fixture","packages":{}}'):
                first=command.freeze(config)
                self.assertEqual(command.freeze(config),first)
                truth.write_text('changed fixture')
                with self.assertRaisesRegex(ValueError,'new run name'):command.freeze(config)

    def test_empty_predictions_and_maskless_predictions(self):
        empty=SimpleNamespace(xyxy=np.empty((0,4)),class_id=None,confidence=None,mask=None)
        pred=rfdetr_predictions(empty,(10,10,3),3,.3)
        self.assertEqual(pred['masks'].shape,(0,1,10,10))
        pred=identity_predictions(np.zeros((10,10,3),dtype=np.uint8),pred,self.encoder,
            dict(score_threshold=.3,mask_threshold=.5,minimum_mask_pixels=1))
        self.assertEqual(pred['embeddings'].shape,(0,4))
        invalid=SimpleNamespace(xyxy=np.zeros((1,4)),class_id=np.array([3]),confidence=np.array([.9]),mask=None)
        with self.assertRaises(ValueError):rfdetr_predictions(invalid,(10,10,3),3,.3)

    def test_new_detector_table_enters_single_scorecard_without_retrieval_claim(self):
        from utils.benchmarking.suite import report,scorecard,command as suite_command
        with TemporaryDirectory() as directory:
            root=Path(directory);base=root/'metrics/test';(base/'detectors').mkdir(parents=True)
            (base/'manifest.yaml').write_text(yaml.safe_dump(dict(protocol=dict(dataset='fixture',frames={'0002':[]}))))
            header='detector,reid,HOTA,DetA,AssA,IDF1,sMOTSA,IDSW,pipeline_ms,video_ms,hardware,recovered,eligible,false_reconnections\n'
            (base/'detectors/summary.csv').write_text(header+'RF-DETR Seg Large,Frozen TransReID,50,40,60,55,45,2,20,25,fixture,1,2,0\n')
            config=dict(output='metrics',run='test',joint_checkpoint='fixture.pt')
            patches=[patch.object(module,'ROOT',root) for module in [report,scorecard,suite_command]]
            for p in patches:p.start();self.addCleanup(p.stop)
            report.render(config,dict(status='complete',stages={'detectors':{'status':'complete'}}))
            ledger=yaml.safe_load((root/'metrics/scorecard/scores.yaml').read_text())
            entries=list(ledger['rows'].values())
            self.assertEqual(len(entries),1)
            self.assertEqual(entries[0]['table'],'detectors')
            self.assertEqual(entries[0]['best']['primary'],50.)
            self.assertEqual(list(root.rglob('*.html')),[root/'milestones.html'])

    def test_interrupted_resume_truncates_uncommitted_rle_and_preserves_ids(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);frames=root/'dataset/training/image_02/0002';frames.mkdir(parents=True)
            labels=root/'dataset/instances_txt';labels.mkdir()
            mask=self.raw()['masks'][0,0].numpy()
            encoded=rle.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode()
            lines=[]
            for index in range(3):
                cv2.imwrite(str(frames/f'{index:06d}.png'),np.zeros((10,10,3),dtype=np.uint8))
                lines.append(f'{index} 1 1 10 10 {encoded}\n')
            (labels/'0002.txt').write_text(''.join(lines))
            config=dict(dataset_root='dataset',output='metrics',state_root='state',run='test',checkpoint_frames=1,
                        score_threshold=.3,mask_threshold=.5,minimum_mask_pixels=1)
            tracking=dict(max_age=15,appearance_weight=.5,min_iou=.05,min_cosine=.7,max_center_distance=2.,
                          max_match_cost=.8,feature_momentum=.9)
            calls=[]
            def interrupted(image):
                calls.append(1)
                if len(calls)==2:raise RuntimeError('fixture interruption')
                return self.raw()
            with self.assertRaises(RuntimeError):
                sequence(root,config,'fixture','0002',interrupted,self.encoder,tracking,'fixed',torch.device('cpu'))
            output=root/'metrics/test/detectors/predictions/fixture/data/0002.txt'
            with output.open('a') as stream:stream.write('uncommitted garbage')
            result=sequence(root,config,'fixture','0002',lambda image:self.raw(),self.encoder,tracking,'fixed',torch.device('cpu'))
            self.assertTrue(result['complete'])
            rows=output.read_text().splitlines()
            self.assertEqual(len(rows),3)
            self.assertEqual([line.split()[1] for line in rows],['1','1','1'])
            sequence(root,config,'fixture','0002',lambda image:self.fail('complete state must be reused'),self.encoder,tracking,'fixed',torch.device('cpu'))
            from utils.full_frame.mots_evaluation import evaluate
            metrics=evaluate(root/'dataset',root/'metrics/test/detectors/predictions',['fixture'],
                             {'0002':3},root/'official_score',print_results=False)[0]
            self.assertAlmostEqual(metrics['HOTA'],100.)
            self.assertAlmostEqual(metrics['IDF1'],100.)

    def test_launcher_keeps_venv_interpreter_and_check_does_not_score(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'configs').mkdir()
            config=yaml.safe_load((command.ROOT/'configs/detector_compare.yaml').read_text())
            config.update(output='metrics',state_root='state')
            for name in ['codino','rfdetr']:
                config[name]['environment']=f'env_{name}'
                path=root/f'env_{name}/bin/python';path.parent.mkdir(parents=True);path.symlink_to('/usr/bin/python3')
            weights=root/config['rfdetr']['checkpoint'];weights.parent.mkdir(parents=True);weights.touch()
            (root/'configs/detector_compare.yaml').write_text(yaml.safe_dump(config))
            with patch.object(command,'ROOT',root),patch.object(command,'execute') as execute,patch.object(command,'freeze') as freeze,patch.object(command,'summarize') as summarize:
                command.main(['--check'])
            self.assertEqual(execute.call_args_list[1].args[0][0],str(root/'env_rfdetr/bin/python'))
            freeze.assert_not_called();summarize.assert_not_called()
            self.assertFalse((root/'metrics'/config['run']/'queue.yaml').exists())


if __name__=='__main__':unittest.main()
