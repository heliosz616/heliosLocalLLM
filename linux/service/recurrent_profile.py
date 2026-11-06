"""Optional sealed host-only recurrent-checkpoint release, with isolated caches."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / 'runtime/recurrent-checkpoints-active.json'


def selected(model):
    if not ACTIVE.exists():
        return None
    record = json.loads(ACTIVE.read_text())['models'].get(model)
    if record is None:
        return None
    path = Path(record['library'])
    if not path.is_absolute() or path.name != 'libllama-server-impl.so':
        raise RuntimeError('Invalid recurrent checkpoint host path')
    if path.stat().st_size != record['bytes']:
        raise RuntimeError('Recurrent checkpoint host size changed')
    # Host libraries are small relative to weights; verify their contents on
    # each activation instead of relying on timestamp-only identity.
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise RuntimeError('Recurrent checkpoint host checksum changed')
    return record


def apply(model, deployment):
    record = selected(model)
    if record is None:
        return deployment
    command, environment, url = deployment
    environment = dict(environment)
    identity = dict(format='native-kv-recurrent-ancestry-v2',
                    base_identity=environment['LLAMA_CACHE_SSD_ID'],
                    host_sha256=record['sha256'], spacing=32768, maximum_checkpoints=6)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(environment['LLAMA_CACHE_SSD_DIR']) / digest
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    (directory / 'identity.manifest').write_text(json.dumps(identity, indent=2) + '\n')
    environment.update(LLAMA_RECURRENT_CHECKPOINTS='1', LLAMA_CACHE_SSD_ID=digest,
                       LD_LIBRARY_PATH=str(Path(record['library']).parent) + ':' + environment['LD_LIBRARY_PATH'])
    environment.pop('LLAMA_RECURRENT_CHECKPOINT_BENCH_CONTROL', None)
    environment.pop('LLAMA_CHECKPOINT_TEST_STOP_FILE', None)
    return command, environment, url
