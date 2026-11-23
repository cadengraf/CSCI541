"""User-launched RF-DETR box/mask fine-tuning and separate fixed-pipeline benchmark."""
import argparse
from copy import deepcopy
import fcntl
from importlib.metadata import version
from pathlib import Path
import tempfile
import yaml
from utils.benchmarking.suite.command import digest,execute,write

ROOT=Path(__file__).resolve().parents[3]


def read(path):
    config=yaml.safe_load((ROOT/path).read_text())
    from .data import check_splits
    check_splits(config)
    if config['device']!='auto':raise ValueError('device must be auto')
    return config


def native_configs(config,comparison,initial,output):
    from rfdetr.config import RFDETRSegLargeConfig,SegmentationTrainConfig
    mc=RFDETRSegLargeConfig(model_name='RFDETRSegLarge',num_classes=1,
        pretrain_weights=str(initial),resolution=comparison['rfdetr']['resolution'],gradient_checkpointing=True)
    tc=SegmentationTrainConfig(**config['training'],dataset_dir=str(ROOT/config['data_output']),
        dataset_file='roboflow',output_dir=str(output),class_names=['car'],accelerator='auto',devices=1)
    return mc,tc


def latest_resume(output):
    import torch
    candidates=[]
    for name in ['last.ckpt','step-last.ckpt']:
        path=output/name
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            if not saved.get('optimizer_states'):raise ValueError('Resume checkpoint lacks optimizer state')
            candidates.append((int(saved['global_step']),path));del saved
    return max(candidates,key=lambda item:item[0])[1] if candidates else None


def worker(config,check=False):
    import torch
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rfdetr.training import RFDETRDataModule,RFDETRModelModule,build_trainer
    from .data import prepare as prepare_data
    from .initialization import prepare as prepare_initial
    comparison=yaml.safe_load((ROOT/config['base_comparison']).read_text())
    if version('rfdetr')!=comparison['rfdetr']['package_version']:raise ValueError('RF-DETR version changed')
    if not torch.cuda.is_available():raise RuntimeError('device:auto selected CPU; use the CUDA training host')
    audit=prepare_data(config,ROOT)
    initial=prepare_initial(config,comparison,ROOT)
    output=ROOT/config['output'];metrics=ROOT/config['metrics'];metrics.mkdir(parents=True,exist_ok=True)
    seed_everything(config['training']['seed'],workers=True)
    if check:
        with tempfile.TemporaryDirectory(prefix='rfdetr-training-check-') as directory:
            smoke=deepcopy(config);smoke['training'].update(num_workers=0,use_ema=False)
            mc,tc=native_configs(smoke,comparison,initial,Path(directory))
            module=RFDETRModelModule(mc,tc);data=RFDETRDataModule(mc,tc)
            trainer=build_trainer(tc,mc,include_training_callbacks=False,fast_dev_run=True,
                logger=False,enable_checkpointing=False,enable_model_summary=False)
            trainer.fit(module,datamodule=data)
            assert trainer.global_step==1
            assert all(torch.isfinite(p).all() for p in module.parameters())
            # Exercise the pinned native inference-checkpoint format and the
            # actual comparison adapter, not a hand-written approximation.
            from rfdetr.training.callbacks.best_model import BestModelCallback
            state=module.model.state_dict()
            args,name,model_config=BestModelCallback._resolve_model_description(trainer,module,state)
            payload=BestModelCallback._build_checkpoint_payload(state,args,trainer,
                model_name=name,model_config_dict=model_config)
            exported=Path(directory)/'inference.pth';torch.save(payload,exported)
            probe=deepcopy(comparison)
            probe['rfdetr'].update(checkpoint=str(exported),checkpoint_format='finetuned',sha256=digest(exported),coco_car_id=0)
            from utils.experiments.detector_compare.models import Detector
            import cv2
            detector=Detector('rfdetr',probe,ROOT,torch.device('cuda'))
            image=next((ROOT/config['data_output']/'valid').glob('*.png'))
            prediction=detector(cv2.imread(str(image)))
            assert prediction['masks'].ndim==4
            print('CHECK ONLY: one real optimizer/validation batch and native export/reload passed; no production checkpoint or scorecard written.',flush=True)
        return
    output.mkdir(parents=True,exist_ok=True)
    files=[*Path(__file__).parent.glob('*.py')]
    signature=dict(config=config,base_comparison=comparison,audit=audit,initial_sha256=digest(initial),
        implementation={str(p.relative_to(ROOT)):digest(p) for p in files},
        packages={name:version(name) for name in ['rfdetr','torch','torchvision','pytorch_lightning','torchmetrics','numpy']})
    path=metrics/'manifest.yaml'
    if path.exists() and yaml.safe_load(path.read_text())!=signature:
        raise ValueError('Training recipe/data/implementation changed; use new output/data/metrics paths')
    if not path.exists():write(path,signature)
    queue=metrics/'queue.yaml'
    if queue.exists() and yaml.safe_load(queue.read_text()).get('status')=='complete':
        finished=yaml.safe_load(queue.read_text())
        if digest(ROOT/finished['selected_checkpoint'])!=finished['selected_sha256']:
            raise ValueError('Completed training checkpoint changed')
        print('RF-DETR training already complete. Next: python benchmark.py rfdetr',flush=True);return
    mc,tc=native_configs(config,comparison,initial,output)
    module=RFDETRModelModule(mc,tc);data=RFDETRDataModule(mc,tc)
    periodic=ModelCheckpoint(dirpath=output,filename='step-last',every_n_train_steps=config['checkpoint_steps'],
        every_n_epochs=0,save_top_k=1,save_on_train_epoch_end=False,enable_version_counter=False,auto_insert_metric_name=False)
    trainer=build_trainer(tc,mc,extra_callbacks=[periodic],enable_model_summary=False,log_every_n_steps=10)
    resume=latest_resume(output)
    write(queue,dict(status='running',resume=str(resume) if resume else None))
    try:
        trainer.fit(module,datamodule=data,ckpt_path=str(resume) if resume else None)
        best=output/'checkpoint_best_total.pth'
        if not best.exists():raise RuntimeError('Training completed without a selected inference checkpoint')
        write(queue,dict(status='complete',global_step=trainer.global_step,selected_checkpoint=str(best.relative_to(ROOT)),
            selected_sha256=digest(best),selection='Internal development-sequence mask mAP; no official validation fitting'))
        # Keep the selected weights and final full resume state; discard redundant
        # native exports only after successful completion.
        for name in ['checkpoint_best_regular.pth','checkpoint_best_ema.pth','last.pth','last_ema.pth','step-last.ckpt']:
            p=output/name
            if p.exists() and (output/'last.ckpt').exists():p.unlink()
        print('Training complete. Next user command: python -u benchmark.py rfdetr',flush=True)
    except BaseException as error:
        write(queue,dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error)))
        raise


def benchmark_config(config):
    metrics=ROOT/config['metrics'];queue=metrics/'queue.yaml'
    status=yaml.safe_load(queue.read_text()) if queue.exists() else {}
    if status.get('status')!='complete':raise ValueError('Complete `python train.py rfdetr` before benchmarking')
    best=ROOT/status['selected_checkpoint']
    if digest(best)!=status['selected_sha256']:raise ValueError('Selected training checkpoint changed')
    comparison=yaml.safe_load((ROOT/config['base_comparison']).read_text())
    comparison['run']=config['run']+'_comparison'
    comparison['rfdetr'].update(checkpoint=str(best.relative_to(ROOT)),checkpoint_format='finetuned',
        sha256=status['selected_sha256'],coco_car_id=0,
        training='COCO initialized; KITTI training subset; internal sequence-dev mask mAP selection')
    path=metrics/'comparison.yaml';write(path,comparison);return path


def main(argv=None,evaluate=False):
    parser=argparse.ArgumentParser(description='Repeat the frozen specialist/tracker benchmark with the completed RF-DETR fine-tuning run.' if evaluate else __doc__)
    parser.add_argument('--config',default='configs/rfdetr_finetune.yaml')
    parser.add_argument('--check',action='store_true',help='Two-frame inference check only' if evaluate else 'One-batch integration check; no production checkpoints')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv);config=read(args.config)
    if args.worker:return worker(config,args.check)
    if evaluate:
        from utils.experiments.detector_compare.command import main as compare,freeze
        path=benchmark_config(config)
        fine=yaml.safe_load(path.read_text())
        baseline=yaml.safe_load((ROOT/config['base_comparison']).read_text())
        original_path=ROOT/baseline['output']/baseline['run']/'manifest.yaml'
        if not original_path.exists():raise ValueError('Complete the original detector comparison first')
        original=yaml.safe_load(original_path.read_text());current=freeze(fine)
        if current['protocol']!=original['protocol'] or current['detectors']['codino']!=original['detectors']['codino']:
            raise ValueError('Frozen test inputs, specialist, tracker or Co-DINO control differ from the original comparison')
        return compare(['--config',str(path.relative_to(ROOT))]+(['--check'] if args.check else []))
    metrics=ROOT/config['metrics'];metrics.mkdir(parents=True,exist_ok=True)
    with (metrics/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('RF-DETR fine-tuning already running') from None
        python=(ROOT/config['environment']).resolve()/'bin/python'
        command=[str(python),'-u','-m','utils.experiments.rfdetr_finetune.command','--config',args.config,'--worker']
        if args.check:command.append('--check')
        execute(command,metrics/('check.log' if args.check else 'training.log'))


def benchmark(argv=None):return main(argv,evaluate=True)


if __name__=='__main__':main()
