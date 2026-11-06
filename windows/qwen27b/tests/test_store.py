import sys
from pathlib import Path
import pytest
import store as store_module
import hashlib
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from store import Store

@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path,dict(cache_max_bytes=100,session_max_bytes=60,session_ttl_seconds=3600))
    yield s
    s.db.close()

def commit(store,sid,payload=b'checkpoint',version='v1'):
    (store.root/'write.tmp').write_bytes(payload)
    store.commit(sid,'write.tmp',version,{'metrics':{'cached_tokens':0}})

def test_restart_and_corruption_fail_closed(store):
    commit(store,'A')
    original = store.get('A')
    store.recover()
    assert store.restore_candidate('A','v1')[1] == 'snapshot_verified'
    (store.root/original['filename']).write_bytes(b'corruption')
    assert store.restore_candidate('A','v1') == (None,'restore_failed_integrity')

def test_atomic_old_snapshot_survives_failed_commit(store):
    commit(store,'A')
    original = store.get('A')
    with pytest.raises(ValueError):
        commit(store,'A',b'x'*61)
    assert store.get('A') == original
    assert store.restore_candidate('A','v1')[1] == 'snapshot_verified'

def test_version_mismatch_does_not_restore(store):
    commit(store,'A')
    assert store.restore_candidate('A','new') == (None,'version_mismatch')

def test_lru_and_orphan_recovery(store):
    commit(store,'A',b'a'*40)
    commit(store,'B',b'b'*40)
    commit(store,'C',b'c'*40)
    assert store.get('A') is None
    assert store.get('B') is not None
    (store.root/'unfinished.tmp').write_bytes(b'partial')
    (store.root/'orphan.bin').write_bytes(b'orphan')
    store.recover()
    assert not (store.root/'unfinished.tmp').exists()
    assert not (store.root/'orphan.bin').exists()
    assert store.restore_candidate('B','v1')[1] == 'snapshot_verified'

def test_commit_replaces_only_own_session(store):
    commit(store,'A',b'aaaa')
    commit(store,'B',b'bbbb')
    old = store.get('A')['filename']
    commit(store,'A',b'new')
    assert not (store.root/old).exists()
    assert (store.root/store.get('B')['filename']).read_bytes() == b'bbbb'

def test_new_snapshots_never_hash_kv_on_commit_or_hit(store,monkeypatch):
    def forbidden(*args):
        raise AssertionError('Hot path must not read the full KV file')
    monkeypatch.setattr(store_module,'sha256',forbidden)
    commit(store,'A')
    assert store.get('A')['digest'].startswith('text:')
    for _ in range(3):
        assert store.restore_candidate('A','v1')[1]=='snapshot_verified'

def test_legacy_snapshot_hash_is_only_checked_once(store,monkeypatch):
    commit(store,'A')
    row=store.get('A')
    metadata=row['metadata'];metadata.pop('snapshot_stat')
    digest=hashlib.sha256(b'checkpoint').hexdigest()
    with store.db:
        store.db.execute('UPDATE sessions SET digest=?,metadata=? WHERE id=?',(digest,json.dumps(metadata),'A'))
    real_sha=store_module.sha256
    calls=[]
    def counted(path):
        calls.append(path)
        return real_sha(path)
    monkeypatch.setattr(store_module,'sha256',counted)
    assert store.restore_candidate('A','v1')[1]=='snapshot_verified'
    assert store.restore_candidate('A','v1')[1]=='snapshot_verified'
    assert len(calls)==1
