"""Pinned source and resumable, SHA-256-verified official teacher weights."""

import hashlib
from pathlib import Path
import subprocess
import time
import requests

from .settings import read, resolve


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def source(settings, root):
    location = resolve(root, settings['source']['directory'])
    if not location.exists():
        location.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', settings['source']['repository'], str(location)], check=True)
        subprocess.run(['git', '-C', str(location), 'checkout', settings['source']['revision']], check=True)
    revision = subprocess.check_output(['git', '-C', str(location), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != settings['source']['revision']:
        raise ValueError(f'Unexpected teacher source revision at {location}: {revision}')
    return location


def weights(settings, root):
    spec = settings['pretrained']
    destination = resolve(root, spec['path'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256(destination) != spec['sha256']:
            raise ValueError(f'Checkpoint SHA-256 mismatch: {destination}')
        return destination
    temporary = destination.with_suffix(destination.suffix + '.partial')
    if temporary.exists() and temporary.stat().st_size >= spec['bytes']:
        if temporary.stat().st_size == spec['bytes'] and sha256(temporary) == spec['sha256']:
            temporary.replace(destination)
            return destination
        temporary.unlink()
    for attempt in range(8):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with requests.get(spec['url'], params={'download': 'true'}, headers=headers,
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                resume = response.status_code == 206 and offset > 0
                if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected range response for checkpoint resume')
                received = offset if resume else 0
                progress = received
                with temporary.open('ab' if resume else 'wb') as file:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        file.write(chunk)
                        received += len(chunk)
                        if received - progress >= 128 * 1024 * 1024:
                            print(f'Teacher weights: {received / spec["bytes"]:.0%}', flush=True)
                            progress = received
            break
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))
    if temporary.stat().st_size != spec['bytes'] or sha256(temporary) != spec['sha256']:
        raise ValueError(f'Incomplete or corrupt teacher download: {temporary}')
    temporary.replace(destination)
    print(f'Verified official teacher checkpoint: {destination}', flush=True)
    return destination


def prepare():
    settings, root = read()
    source(settings, root)
    weights(settings, root)
