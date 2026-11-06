"""Publication adapter checks, using only temporary synthetic artifacts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import profiles
from settings import load_json


def install_profile(tmp_path, monkeypatch):
    executable = tmp_path / 'engine'
    weights = tmp_path / 'weights'
    executable.write_bytes(b'synthetic-executable')
    weights.write_bytes(b'synthetic-weights')
    value = {'command': [str(executable), '--model', str(weights), '--alias', 'placeholder'],
             'url': 'http://127.0.0.1:17801', 'environment': {},
             'artifacts': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in (executable, weights)]}
    path = tmp_path / 'profiles.local.json'
    def write():
        path.write_text(json.dumps({'models': {profiles.Q5: value}}))
    write()
    monkeypatch.setattr(profiles, 'PROFILES', path)
    return value, write, weights


def test_artifact_tampering_fails(tmp_path, monkeypatch):
    _, _, weights = install_profile(tmp_path, monkeypatch)
    profiles.verify_artifacts(profiles.Q5, tmp_path)
    weights.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        profiles.verify_artifacts(profiles.Q5, tmp_path)


def test_unsealed_required_model_fails(tmp_path, monkeypatch):
    value, write, _ = install_profile(tmp_path, monkeypatch)
    value['artifacts'].pop(); write()
    with pytest.raises(RuntimeError, match='cover --model'):
        profiles.verify_artifacts(profiles.Q5, tmp_path)


def test_backend_must_be_loopback(tmp_path, monkeypatch):
    value, write, _ = install_profile(tmp_path, monkeypatch)
    value['url'] = 'http://example.com:17801'; write()
    with pytest.raises(ValueError, match='loopback'):
        profiles.configuration(profiles.Q5)


def test_private_diagnostics_cannot_leak_into_runtime(tmp_path, monkeypatch):
    value, write, _ = install_profile(tmp_path, monkeypatch)
    value['environment'] = {'LLAMA_CHECKPOINT_TEST_STOP_FILE': 'synthetic-stop',
                            'LLAMA_RECURRENT_CHECKPOINT_BENCH_CONTROL': 'synthetic-toggle'}
    write()
    command, env, url = profiles.profile(profiles.Q5, tmp_path / 'state')
    assert command[-1] == profiles.Q5
    assert 'LLAMA_CHECKPOINT_TEST_STOP_FILE' not in env
    assert 'LLAMA_RECURRENT_CHECKPOINT_BENCH_CONTROL' not in env


def test_settings_expand_paths_without_shell_execution(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'state': '${ROOT}/state', 'literal': '$(synthetic-command)'}))
    loaded = load_json(path, root=tmp_path)
    assert loaded == {'state': str(tmp_path / 'state'), 'literal': '$(synthetic-command)'}


def test_initializer_preserves_existing_key_and_uses_private_permissions(tmp_path):
    source = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('archive_configure', source / 'configure.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for name in ('config', 'engine-profiles'):
        (tmp_path / (name + '.example.json')).write_bytes((source / (name + '.example.json')).read_bytes())
    module.initialize(tmp_path)
    key = tmp_path / 'state/serving/api-key'
    original = key.read_bytes()
    module.initialize(tmp_path)
    assert key.read_bytes() == original
    assert key.stat().st_mode & 0o777 == 0o600
    assert key.parent.stat().st_mode & 0o777 == 0o700
