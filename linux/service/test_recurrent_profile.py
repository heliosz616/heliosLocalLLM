import hashlib
import json

import pytest
import recurrent_profile


def test_optional_release_is_inactive_until_explicitly_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(recurrent_profile, 'ACTIVE', tmp_path / 'active.json')
    deployment = (['server'], {}, 'http://localhost')
    assert recurrent_profile.apply('qwen3.8-flash', deployment) is deployment


def test_release_verifies_host_and_isolates_model_cache(tmp_path, monkeypatch):
    active = tmp_path / 'active.json'
    monkeypatch.setattr(recurrent_profile, 'ACTIVE', active)
    library = tmp_path / 'libllama-server-impl.so'
    library.write_bytes(b'host-only-test-fixture')
    active.write_text(json.dumps(dict(models={'qwen3.8-flash': dict(
        library=str(library), bytes=library.stat().st_size,
        sha256=hashlib.sha256(library.read_bytes()).hexdigest())})))
    env = dict(LLAMA_CACHE_SSD_DIR=str(tmp_path / 'cache'), LLAMA_CACHE_SSD_ID='a' * 64,
               LD_LIBRARY_PATH='/original/inference/libs', LLAMA_RECURRENT_CHECKPOINT_BENCH_CONTROL='/private/test',
               LLAMA_CHECKPOINT_TEST_STOP_FILE='/private/stop')
    command, result, url = recurrent_profile.apply('qwen3.8-flash', (['server'], env, 'http://localhost'))
    assert result['LLAMA_CACHE_SSD_ID'] != env['LLAMA_CACHE_SSD_ID']
    assert result['LLAMA_RECURRENT_CHECKPOINTS'] == '1'
    assert result['LD_LIBRARY_PATH'].endswith(':/original/inference/libs')
    assert 'LLAMA_RECURRENT_CHECKPOINT_BENCH_CONTROL' not in result
    assert 'LLAMA_CHECKPOINT_TEST_STOP_FILE' not in result
    assert 'LLAMA_RECURRENT_CHECKPOINTS' not in env
    library.write_bytes(b'X' * library.stat().st_size)
    with pytest.raises(RuntimeError, match='checksum'):
        recurrent_profile.apply('qwen3.8-flash', (['server'], env, url))
