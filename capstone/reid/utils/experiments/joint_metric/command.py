"""Train the two matched arms, then benchmark them with a separate user command."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import yaml
from utils.benchmarking.suite.command import execute, write

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('control', 'candidate')


def benchmark_configs():
    """Keep reference experiments frozen; evaluate only the changed joint paths."""
    base = yaml.safe_load((ROOT/'configs/benchmark_suite.yaml').read_text())
    result = []
    for arm in ARMS:
        config = deepcopy(base)
        config.update(run=f'joint_metric_{arm}_v1',
            joint_checkpoint=f'checkpoints/joint_metric/{arm}.best.pt',
            joint_last_checkpoint=f'checkpoints/joint_metric/{arm}.last.pt')
        config['inputs'] = ['checkpoints/reid_teacher/selected.yaml', 'checkpoints/reid_teacher/transreid.best.pt',
                            'data/aerial/vrai/*.csv']
        for stage in config['stages']:
            stage['enabled'] = stage['id'] in {'joint','recovery','joint_recovery'}
        path = ROOT/'metrics/joint_metric'/f'benchmark_{arm}.yaml'
        write(path, config)
        result.append(path)
    return result


def main(argv=None, evaluate=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if evaluate:parser.error('Benchmark workers are managed by the suite')
        from .model import MetricJoint
        from .settings import read
        from utils.experiments.reid_balance import training
        # Process-local injection: do not edit signed historical training modules.
        training.CoDINOJoint = MetricJoint
        return training.train(args.worker, settings_reader=read)
    output = ROOT/'metrics/joint_metric'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'queue.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Joint metric experiment is already running') from None
        if evaluate:
            queue=output/'queue.yaml'
            if not queue.is_file() or yaml.safe_load(queue.read_text()).get('status')!='complete':
                raise ValueError('Finish both arms with `python train.py joint-metric` before benchmarking')
            from utils.benchmarking.suite.command import main as suite
            for path in benchmark_configs():
                suite(['--config',str(path.relative_to(ROOT))])
            return
        teacher=yaml.safe_load((ROOT/'configs/teacher.yaml').read_text())
        python=(ROOT/teacher['environment']/'bin/python').resolve()
        if not python.is_file():raise FileNotFoundError(python)
        status=dict(status='running',completed=[],current=None)
        try:
            for arm in ARMS:
                status['current']=arm;write(output/'queue.yaml',status)
                print(f'Joint metric training: {arm}; logs: {output/(arm+".log")}',flush=True)
                execute([str(python),'-u','-m','utils.experiments.joint_metric.command','--worker',arm], output/f'{arm}.log')
                status['completed'].append(arm)
            status.update(status='complete',current=None)
            benchmark_configs()
            print('Training complete. Next user-launched command: python benchmark.py joint-metric',flush=True)
        except BaseException as error:
            status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            raise
        finally:write(output/'queue.yaml',status)


def benchmark(argv=None):
    return main(argv,evaluate=True)


if __name__=='__main__':main()
