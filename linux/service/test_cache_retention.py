import json
import os
from pathlib import Path

import pytest

from cache_retention import cache_roots, prune_snapshots


NOW = 200000.0


def test_pending_reservation_protects_commit_and_partial_from_maintenance(tmp_path):
    from cache_retention import enforce_capacity
    meta, payload = snapshot(tmp_path, age=100000)
    reservation = meta.with_suffix('.reserve')
    reservation.write_text('{}')
    partial = meta.with_suffix('.part')
    partial.write_bytes(b'pending')
    os.utime(partial, (1, 1))
    lock = tmp_path / '.kv-quota.lock'
    prune_snapshots([tmp_path], now=NOW, opened=set(), lock_path=lock)
    enforce_capacity([tmp_path], 1, opened=set(), lock_path=lock)
    assert all(p.exists() for p in (meta, payload, partial, reservation))
    reservation.unlink()
    prune_snapshots([tmp_path], now=NOW, opened=set(), lock_path=lock)
    assert not meta.exists() and not payload.exists() and not partial.exists()


def snapshot(tmp_path, *, age, payload_age=None, compressed=False):
    namespace = tmp_path / ('a' * 64)
    namespace.mkdir(exist_ok=True)
    meta = namespace / '123-1.json'
    payload = namespace / ('123-1.bin.zst' if compressed else '123-1.bin')
    meta.write_text(json.dumps(dict(version=1, identity=namespace.name, bytes=7)))
    payload.write_bytes(b'payload')
    os.utime(meta, (NOW - age, NOW - age))
    stamp = NOW - (age if payload_age is None else payload_age)
    os.utime(payload, (stamp, stamp))
    return meta, payload


def test_last_reuse_protects_old_payload(tmp_path):
    meta, payload = snapshot(tmp_path, age=5, payload_age=100000)
    result = prune_snapshots([tmp_path], now=NOW, opened=set())
    assert meta.exists() and payload.exists()
    assert result['retained_snapshots'] == 1


def test_explicit_clear_removes_recent_snapshot(tmp_path):
    meta, payload = snapshot(tmp_path, age=5)
    result = prune_snapshots([tmp_path], now=NOW, opened=set(), clear=True)
    assert not meta.exists() and not payload.exists()
    assert result['explicit_clear']


def test_expiry_removes_pair_but_preserves_history_and_identity(tmp_path):
    meta, payload = snapshot(tmp_path, age=86400)
    history = tmp_path / 'responses.sqlite3'
    history.write_bytes(b'history')
    identity = meta.parent / 'identity.manifest'
    identity.write_text('identity')
    result = prune_snapshots([tmp_path], now=NOW, opened=set())
    assert not meta.exists() and not payload.exists()
    assert history.read_bytes() == b'history' and identity.exists()
    assert len(result['deleted_files']) == 2


def test_compression_does_not_reset_expiry(tmp_path):
    meta, payload = snapshot(tmp_path, age=100000, payload_age=10, compressed=True)
    prune_snapshots([tmp_path], now=NOW, opened=set())
    assert not meta.exists() and not payload.exists()


def test_open_snapshot_is_retained_atomically(tmp_path):
    meta, payload = snapshot(tmp_path, age=100000)
    result = prune_snapshots([tmp_path], now=NOW, opened={str(payload.absolute())})
    assert meta.exists() and payload.exists()
    assert result['skipped_open'] == 1


def test_orphan_expiry_preserves_recent_writers_and_foreign_files(tmp_path):
    parent = tmp_path / ('a' * 64)
    parent.mkdir()
    old = parent / '100-1.part'
    new = parent / '100-2.part'
    other = parent / 'weights.bin'
    for p in [old, new, other]:
        p.write_bytes(b'x')
    os.utime(old, (0, 0))
    os.utime(other, (0, 0))
    prune_snapshots([tmp_path], now=NOW, opened=set())
    assert not old.exists() and new.exists() and other.exists()


def test_symlinks_and_foreign_metadata_are_preserved(tmp_path):
    meta, payload = snapshot(tmp_path, age=100000)
    meta.write_text(json.dumps(dict(version=1, identity='b' * 64)))
    os.utime(meta, (0, 0))
    external = tmp_path / 'outside'
    external.write_text('keep')
    link = meta.parent / '124-1.bin'
    link.symlink_to(external)
    prune_snapshots([tmp_path], now=NOW, opened=set())
    assert meta.exists() and payload.exists() and link.is_symlink()
    assert external.read_text() == 'keep'


def test_custom_state_does_not_touch_project_test_caches(tmp_path):
    roots = cache_roots(tmp_path / 'custom', tmp_path)
    assert {p.name for p in roots} == {'q5-ssd-cache', 'flash-next-ssd-cache'}
    assert all(p.parent == tmp_path / 'custom' for p in roots)


@pytest.mark.parametrize('seconds', [0, -1, float('inf'), float('nan')])
def test_invalid_retention_is_rejected(tmp_path, seconds):
    with pytest.raises(ValueError):
        prune_snapshots([tmp_path], seconds, now=NOW, opened=set())


def test_capacity_evicts_oldest_across_models_and_keeps_hot_gpu_uninvolved(tmp_path):
    from cache_retention import cache_bytes, enforce_capacity
    old_meta, old_payload = snapshot(tmp_path, age=100)
    other = tmp_path / 'other'
    other.mkdir()
    new_meta, new_payload = snapshot(other, age=5)
    newest_size = cache_bytes([other])
    report = enforce_capacity([tmp_path], newest_size, opened=set())
    assert not old_meta.exists() and not old_payload.exists()
    assert new_meta.exists() and new_payload.exists()
    assert report['after_bytes'] == newest_size


def test_capacity_protects_open_snapshot(tmp_path):
    from cache_retention import enforce_capacity
    meta, payload = snapshot(tmp_path, age=5)
    result = enforce_capacity([tmp_path], 1, opened={str(payload.resolve())})
    assert meta.exists() and payload.exists()
    assert result['after_bytes'] > 1 and result['deleted_files'] == []
