"""Run/resume the configured benchmark suite, or render existing results only."""
import argparse
from copy import deepcopy
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
import yaml

ROOT = Path(__file__).resolve().parents[3]
ALLOWED = {'joint', 'aerial', 'aerial-transfer', 'vehicle-reference',
           'vehicle-recovery', 'joint-recovery', 'cityflow'}
IDS = dict(zip(['joint', 'aerial', 'aerial-transfer', 'vehicle-reference',
                'vehicle-recovery', 'joint-recovery', 'cityflow'],
               ['joint', 'aerial', 'transfer', 'reference', 'recovery', 'joint_recovery', 'cityflow']))
REQUIRED = {
    'joint': ['vrai.csv', 'kitti_mots/summary.csv', 'timing.csv'],
    'aerial': ['report.yaml'], 'aerial-transfer': ['summary.csv'],
    'vehicle-reference': ['summary.md', 'original.yaml', 'mixed_adasp.yaml', 'msinet.yaml'],
    'vehicle-recovery': ['report.yaml', 'gate.yaml', 'comparison.mp4'],
    'joint-recovery': ['report.yaml', 'gate.yaml', 'comparison.mp4'],
    'cityflow': ['report.yaml', 'review.mp4'],
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(yaml.safe_dump(data, sort_keys=False))
    temporary.replace(path)


def read_config(path):
    config = yaml.safe_load((ROOT/path).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', config['run']):
        raise ValueError('run must be a single folder name')
    if config['device'] != 'auto':
        raise ValueError('device must be auto')
    seen = set()
    for stage in config['stages']:
        if stage['id'] in seen or not re.fullmatch(r'[a-z0-9_]+', stage['id']):
            raise ValueError('Stage IDs must be unique safe folder names')
        if stage['command'] not in ALLOWED:
            raise ValueError('Only supported evaluation commands are allowed; no training or shell commands')
        if stage['id'] != IDS[stage['command']]:
            raise ValueError('Keep the documented stage IDs so report sections and dependency paths match')
        if not set(stage.get('depends', [])) <= seen:
            raise ValueError('Dependencies must appear before their consumer')
        seen.add(stage['id'])
    return config


def fingerprint(config):
    files = [*ROOT.glob('*.py'), *ROOT.joinpath('configs').glob('*.yaml')]
    files += [p for p in ROOT.joinpath('utils').rglob('*.py') if 'vendor' not in p.parts]
    patterns = [config['joint_checkpoint'], config['joint_last_checkpoint'], *config.get('inputs', [])]
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(str(ROOT/pattern)) if Path(p).is_file()]
        if not matches:
            raise FileNotFoundError(f'No suite input matches {pattern}')
        files.extend(matches)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}
    environment = dict(python=sys.version, executable=str(Path(sys.executable).resolve()))
    for package in ['torch', 'torchvision', 'numpy', 'Pillow']:
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = 'not installed'
    teacher = yaml.safe_load((ROOT/'configs/teacher.yaml').read_text()) if (ROOT/'configs/teacher.yaml').exists() else {}
    for p in (ROOT/teacher.get('environment', '.missing-env')).glob('lib/python*/site-packages/torch/version.py'):
        environment['teacher_torch_version_sha256'] = digest(p)
    token = hashlib.sha256(json.dumps(dict(config=config, files=hashes, environment=environment), sort_keys=True).encode()).hexdigest()
    return dict(sha256=token, files=hashes, environment=environment)


def prepare_stage(config, stage):
    """Rebase writable paths and dependencies, never mutate source configs/weights."""
    base = ROOT/config['output']/config['run']
    data = ROOT/config['state_root']/config['run']
    output = base/stage['id']
    if stage['command'] == 'joint':
        settings = dict(checkpoint=config['joint_checkpoint'], last_checkpoint=config['joint_last_checkpoint'],
                        output=str(output.relative_to(ROOT)),
                        predictions_root=str((data/'joint_predictions').relative_to(ROOT)),
                        tracking_name='suite_joint', device='auto')
    elif stage['command'] == 'joint-recovery':
        settings = dict(joint_config=str((base/'configs/joint.yaml').relative_to(ROOT)),
                        source_config=str((base/'configs/recovery.yaml').relative_to(ROOT)),
                        output=str(output.relative_to(ROOT)), state_root=str((data/stage['id']).relative_to(ROOT)),
                        device='auto')
    else:
        settings = deepcopy(yaml.safe_load((ROOT/stage['config']).read_text()))
        settings.update(output=str(output.relative_to(ROOT)), state_root=str((data/stage['id']).relative_to(ROOT)))
        links = {
            'aerial-transfer': {'mixed_selection': base/'aerial/selection.yaml'},
            'vehicle-reference': {'transfer_selection': base/'transfer/selection.pt', 'transfer_state': data/'transfer'},
            'cityflow': {'reference_selection': base/'reference/selection.pt', 'source_thresholds': base/'reference/thresholds.yaml'},
        }
        settings.update({key: str(path.relative_to(ROOT)) for key, path in links.get(stage['command'], {}).items()})
    path = base/'configs'/f'{stage["id"]}.yaml'
    write(path, settings)
    return [sys.executable, '-u', str(ROOT/'benchmark.py'), stage['command'], '--config', str(path.relative_to(ROOT))]


def execute(command, log):
    """Stream logs to both tmux and disk; stop the entire child tree on Ctrl+C."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', buffering=1) as stream:
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                        process.wait(timeout=10)
                        break
                    except ProcessLookupError:
                        break
                    except subprocess.TimeoutExpired:
                        continue
            raise
        finally:
            process.stdout.close()


def evidence(directory):
    # Metrics/config records only, not videos, feature banks, weights or logs.
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.suffix in {'.csv', '.yaml'} and p.name != 'queue.yaml'}


def missing_results(stage, directory):
    return [name for name in REQUIRED[stage['command']] if not (directory/name).is_file()]


def discard_duplicate_pages(directory):
    for path in directory.rglob('*.html'):
        path.unlink()


def release_completed_state(config):
    if config.get('retain_completed_state', False):
        return
    path = ROOT/config['state_root']/config['run']
    safe = ROOT/'data/benchmark_suite'/config['run']
    if path != safe or path.is_symlink():
        print('State retained: automatic cleanup is limited to data/benchmark_suite/<run>.', flush=True)
        return
    if path.exists():
        predictions=path/'joint_predictions'
        if predictions.exists():
            destination=ROOT/config['output']/config['run']/'joint/predictions'
            if destination.exists():
                print('Scratch retained: an archived prediction folder already exists.', flush=True)
                return
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(predictions),destination)
        shutil.rmtree(path)
        print('Removed completed suite feature/replay caches; metric records and videos retained.', flush=True)


def run(config):
    from .report import render
    base = ROOT/config['output']/config['run']
    base.mkdir(parents=True, exist_ok=True)
    with (base/'suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This suite is already running') from None
        print('Checking suite source/config/checkpoint fingerprints…', flush=True)
        signature = fingerprint(config)
        path = base/'suite.yaml'
        state = yaml.safe_load(path.read_text()) if path.exists() else dict(
            signature=signature, config=config, started=datetime.now(timezone.utc).isoformat(), stages={})
        if state['signature'] != signature:
            raise ValueError('Suite inputs changed. Set a new run name in configs/benchmark_suite.yaml; old results remain intact.')
        state['status'] = 'running'
        write(path, state)
        try:
            for stage in config['stages']:
                name = stage['id']
                previous = state['stages'].get(name, {})
                if not stage.get('enabled', True):
                    state['stages'][name] = dict(status='disabled')
                elif any(state['stages'].get(dep, {}).get('status') != 'complete' for dep in stage.get('depends', [])):
                    state['stages'][name] = dict(status='blocked', error='An upstream stage is incomplete')
                elif previous.get('status') == 'complete' and previous.get('evidence') and not missing_results(stage, base/name) and previous['evidence'] == evidence(base/name):
                    print(f'{name}: completed stage reused', flush=True)
                else:
                    state['stages'][name] = dict(status='running')
                    write(path, state)
                    render(config, state)
                    try:
                        execute(prepare_stage(config, stage), base/'logs'/f'{name}.log')
                        missing = missing_results(stage, base/name)
                        if missing:
                            raise RuntimeError(f'Evaluator did not produce required results: {missing}')
                        discard_duplicate_pages(base/name)
                        artifacts = evidence(base/name)
                        if not artifacts:
                            raise RuntimeError('Evaluator exited without metric records')
                        state['stages'][name] = dict(status='complete', evidence=artifacts)
                    except Exception as error:
                        state['stages'][name] = dict(status='failed', error=str(error))
                        if not config.get('continue_on_error', True):
                            raise
                write(path, state)
                render(config, state)
            state['status'] = 'complete' if all(s['status'] in {'complete', 'disabled'} for s in state['stages'].values()) else 'incomplete'
        except BaseException as error:
            state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            state['error'] = str(error)
            raise
        finally:
            write(path, state)
            render(config, state)
        if state['status'] != 'complete':
            raise SystemExit('Suite has incomplete stages; see stage logs. Rerun the same command to retry.')
        release_completed_state(config)
        # Archiving official KITTI inputs adds small provenance/timing records.
        for name, record in state['stages'].items():
            if record['status']=='complete':record['evidence']=evidence(base/name)
        write(path,state)
        print(f'Best-model scorecard: {ROOT / "milestones.html"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/benchmark_suite.yaml')
    parser.add_argument('--report-only', action='store_true', help='Render saved tables without running models')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.report_only:
        from .report import render
        path = ROOT/config['output']/config['run']/'suite.yaml'
        render(config, yaml.safe_load(path.read_text()) if path.exists() else dict(status='not started', stages={}))
        return
    run(config)


if __name__ == '__main__':
    main()
