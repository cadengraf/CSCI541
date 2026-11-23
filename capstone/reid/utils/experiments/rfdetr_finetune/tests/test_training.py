from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from pycocotools import mask as rle
import torch
import yaml
from utils.experiments.rfdetr_finetune import command
from utils.experiments.rfdetr_finetune.data import prepare,check_splits
from utils.experiments.rfdetr_finetune.initialization import convert_weights


class TrainingTests(unittest.TestCase):
    def config(self):return command.read('configs/rfdetr_finetune.yaml')

    def test_official_validation_cannot_enter_training_or_selection(self):
        config=self.config();check_splits(config)
        config['development_sequences']=[2,19]
        with self.assertRaises(ValueError):check_splits(config)

    def test_coco_export_has_real_rle_car_zero_empty_frames_and_links_only(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config=self.config();config.update(dataset_root='dataset',data_output='export')
            labels=root/'dataset/instances_txt';labels.mkdir(parents=True)
            mask=np.zeros((8,10),np.uint8);mask[2:6,3:7]=1
            counts=rle.encode(np.asfortranarray(mask))['counts'].decode()
            for number in config['train_sequences']+config['development_sequences']:
                seq=f'{number:04d}';folder=root/'dataset/training/image_02'/seq;folder.mkdir(parents=True)
                for frame in range(2):Image.new('RGB',(10,8)).save(folder/f'{frame:06d}.png')
                (labels/f'{seq}.txt').write_text(f'0 1001 1 8 10 {counts}\n0 2001 2 8 10 {counts}\n')
            (labels/'0002.txt').write_text('Official validation must never be opened by the training exporter')
            audit=prepare(config,root)
            self.assertEqual(audit['splits']['train']['images'],20)
            self.assertEqual(audit['splits']['valid']['images'],4)
            data=json.loads((root/'export/train/_annotations.coco.json').read_text())
            self.assertEqual(len(data['annotations']),10)
            self.assertEqual(data['annotations'][0]['category_id'],0)
            self.assertEqual(data['annotations'][0]['bbox'],[3.,2.,4.,4.])
            self.assertEqual(data['annotations'][0]['area'],16.)
            self.assertTrue(all((root/'export/train'/row['file_name']).is_symlink() for row in data['images']))
            self.assertFalse(any(p.name.startswith('0002_') for p in (root/'export').rglob('*.png')))
            self.assertEqual(prepare(config,root),audit)

    def test_car_logits_preserved_and_query_metadata_is_group_aware(self):
        state=dict({'class_embed.weight':torch.arange(40).reshape(5,8),
                    'class_embed.bias':torch.arange(5),
                    'transformer.enc_out_class_embed.0.weight':torch.arange(40).reshape(5,8),
                    'refpoint_embed.weight':torch.arange(24).reshape(6,4),
                    'query_feat.weight':torch.arange(48).reshape(6,8)})
        converted=convert_weights({'model':state},3,3,2)
        torch.testing.assert_close(converted['model']['class_embed.weight'],state['class_embed.weight'][[3,4]])
        torch.testing.assert_close(converted['model']['transformer.enc_out_class_embed.0.weight'],state['transformer.enc_out_class_embed.0.weight'][[3,4]])
        self.assertEqual(converted['args']['num_queries'],3)
        self.assertEqual(converted['args']['group_detr'],2)
        self.assertEqual(state['class_embed.weight'].shape,(5,8))

    def test_resume_uses_optimizer_progress_and_requires_full_state(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            torch.save(dict(global_step=100,optimizer_states=[{}]),root/'last.ckpt')
            torch.save(dict(global_step=200,optimizer_states=[{}]),root/'step-last.ckpt')
            self.assertEqual(command.latest_resume(root).name,'step-last.ckpt')
            torch.save(dict(global_step=300,optimizer_states=[]),root/'last.ckpt')
            with self.assertRaises(ValueError):command.latest_resume(root)

    def test_benchmark_mapping_and_original_config_are_preserved(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config=self.config();config.update(metrics='metrics',base_comparison='comparison.yaml')
            comparison=yaml.safe_load((command.ROOT/'configs/detector_compare.yaml').read_text())
            (root/'comparison.yaml').write_text(yaml.safe_dump(comparison));before=(root/'comparison.yaml').read_bytes()
            (root/'metrics').mkdir();checkpoint=root/'best.pth';checkpoint.write_bytes(b'fixture')
            status=dict(status='complete',selected_checkpoint='best.pth',selected_sha256=command.digest(checkpoint))
            (root/'metrics/queue.yaml').write_text(yaml.safe_dump(status))
            with patch.object(command,'ROOT',root):path=command.benchmark_config(config)
            generated=yaml.safe_load(path.read_text())
            self.assertEqual(generated['rfdetr']['coco_car_id'],0)
            self.assertEqual(generated['rfdetr']['checkpoint_format'],'finetuned')
            self.assertEqual(generated['codino'],comparison['codino'])
            self.assertEqual(generated['specialist_selection'],comparison['specialist_selection'])
            self.assertEqual((root/'comparison.yaml').read_bytes(),before)


if __name__=='__main__':unittest.main()
