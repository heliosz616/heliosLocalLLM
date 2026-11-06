"""Operator-supplied native engine profiles for the sanitized source archive.

Private deployment manifests and device inventory are deliberately not bundled.
The native backend must implement the archived protocol and state extensions.
"""
from pathlib import Path
import hashlib
import json
import os
import re
from urllib.parse import urlsplit
from settings import ROOT, load_json

Q5 = 'qwen3.8-27b'
FLASH = 'qwen3.8-flash'
ALIASES = {Q5: Q5, FLASH: FLASH, 'q5': Q5, 'flash': FLASH,
           'qwen-swift': Q5, 'qwen-flash-next': FLASH,
           'swift-1.5-q5ks': Q5, 'swift-q5': Q5, 'swift-q5ks': Q5,
           'Swift-1.5-Qwen3.8-27B-Q5_K_S.gguf': Q5}
PROFILES = ROOT / 'engine-profiles.local.json'


def configuration(model):
    if model not in (Q5, FLASH):
        raise ValueError('Unknown or retired model')
    if not PROFILES.exists():
        raise RuntimeError('Configure engine-profiles.local.json before loading a model')
    record = load_json(PROFILES)['models'][model]
    url = urlsplit(record['url'])
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.username or url.password:
        raise ValueError('Native engine must use an unauthenticated loopback URL')
    if not isinstance(record['command'], list) or not record['command'] or not all(isinstance(x, str) for x in record['command']):
        raise ValueError('Native command must be a nonempty argument array')
    return record


def verify_artifacts(model, state):
    record = configuration(model)
    entries = record.get('artifacts', [])
    if not entries:
        raise RuntimeError('List and checksum native executable, libraries, template and model artifacts first')
    checked = set()
    for item in entries:
        expected = item.get('sha256', '')
        if not re.fullmatch('[0-9a-f]{64}', expected or ''):
            raise RuntimeError('Every native artifact requires a SHA-256 checksum')
        path = Path(item['path']).resolve()
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError('Native artifact checksum mismatch: ' + path.name)
        checked.add(path)
    command = record['command']
    if Path(command[0]).resolve() not in checked:
        raise RuntimeError('Native executable is missing from the artifact manifest')
    for flag in ('--model', '--mmproj', '--chat-template-file'):
        if flag in command and Path(command[command.index(flag) + 1]).resolve() not in checked:
            raise RuntimeError('Artifact manifest must cover ' + flag)


def profile(model, state):
    record = configuration(model)
    command = list(record['command'])
    if '--alias' in command:
        command[command.index('--alias') + 1] = model
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('LLAMA_', 'GGML_', 'STRATA_', 'SWIFT_')) and key != 'LD_PRELOAD'}
    env.update(record.get('environment', {}))
    env.pop('LLAMA_RECURRENT_CHECKPOINT_BENCH_CONTROL', None)
    env.pop('LLAMA_CHECKPOINT_TEST_STOP_FILE', None)
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    cache = Path(state) / ('q5-ssd-cache' if model == Q5 else 'flash-next-ssd-cache')
    directory = cache / identity
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    (directory / 'identity.manifest').write_text(json.dumps({'identity': identity, 'model': model}) + '\n')
    env.update(LLAMA_CACHE_SSD_DIR=str(cache), LLAMA_CACHE_SSD_ID=identity,
               LLAMA_CACHE_SSD_BYTES=str(64_000_000_000))
    from recurrent_profile import apply
    return apply(model, (command, env, record['url']))
