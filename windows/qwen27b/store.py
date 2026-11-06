"""Atomic immutable snapshots; text hashes identify prompts, stat checks avoid KV rescans."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
import threading
from functools import wraps

def synchronized(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.mutex:
            return method(self, *args, **kwargs)
    return guarded

def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

class Store:
    def __init__(self, root, config):
        self.mutex = threading.RLock()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.db = sqlite3.connect(self.root / 'sessions.sqlite', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, filename TEXT, digest TEXT, size INTEGER, version TEXT, accessed REAL, metadata TEXT)')
        self.db.commit()
        self.recover()

    @synchronized
    def entries(self):
        return self.db.execute('SELECT id,size,accessed FROM sessions ORDER BY accessed').fetchall()

    @synchronized
    def recover(self):
        referenced = {r[0] for r in self.db.execute('SELECT filename FROM sessions') if r[0]}
        for path in self.root.glob('*.tmp'):
            path.unlink()
        for path in self.root.glob('*.bin'):
            if path.name not in referenced:
                path.unlink()

    @synchronized
    def get(self, sid):
        row = self.db.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone()
        if row is None:
            return None
        keys = ('id','filename','digest','size','version','accessed','metadata')
        result = dict(zip(keys,row))
        result['metadata'] = json.loads(result['metadata'])
        return result

    @synchronized
    def create(self, sid):
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO sessions VALUES (?,NULL,NULL,0,NULL,?,?)', (sid,time.time(),'{}'))
        return self.get(sid)

    @synchronized
    def delete(self, sid):
        row = self.get(sid)
        with self.db:
            self.db.execute('DELETE FROM sessions WHERE id=?',(sid,))
        if row and row['filename']:
            (self.root/row['filename']).unlink(missing_ok=True)

    @synchronized
    def prune(self, incoming=0, protect=None):
        rows = self.db.execute('SELECT id,size,accessed FROM sessions ORDER BY accessed').fetchall()
        total = sum(r[1] for r in rows)
        for sid, size, accessed in rows:
            if sid == protect:
                continue
            if total+incoming > self.config['cache_max_bytes'] or time.time()-accessed > self.config['session_ttl_seconds']:
                self.delete(sid)
                total -= size
        if total+incoming > self.config['cache_max_bytes']:
            raise ValueError('Global snapshot quota exceeded; previous snapshot preserved')

    @synchronized
    def commit(self, sid, temporary, version, metadata):
        path = self.root/temporary
        size = path.stat().st_size
        if size > self.config['session_max_bytes']:
            path.unlink(missing_ok=True)
            raise ValueError('Session snapshot quota exceeded; previous snapshot preserved')
        old = self.get(sid)
        self.prune(incoming=size, protect=sid)
        # Hash small request text/metadata, never scan multi-GB KV on the hot path.
        fingerprint = metadata.get('prompt_fingerprint',{}).get('sha256')
        digest = 'text:'+(fingerprint or hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest())
        with path.open('rb+') as f:
            os.fsync(f.fileno())
        name = uuid.uuid4().hex+'.bin'
        os.replace(path, self.root/name)
        metadata['snapshot_stat'] = {'mtime_ns':(self.root/name).stat().st_mtime_ns}
        # Previous file remains valid until the durable SQLite transaction commits.
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?)',
                            (sid,name,digest,size,version,time.time(),json.dumps(metadata,ensure_ascii=False)))
        if old and old['filename']:
            (self.root/old['filename']).unlink(missing_ok=True)
        return size

    @synchronized
    def restore_candidate(self, sid, version):
        row = self.get(sid)
        if not row or not row['filename']:
            return None, 'cache_miss'
        if time.time()-row['accessed'] > self.config['session_ttl_seconds']:
            self.delete(sid)
            return None, 'expired'
        if row['version'] != version:
            return None, 'version_mismatch'
        path = self.root/row['filename']
        if not path.exists() or path.stat().st_size != row['size']:
            return None, 'restore_failed_integrity'
        current_stat = path.stat()
        recorded_stat = row['metadata'].get('snapshot_stat')
        if recorded_stat:
            if current_stat.st_mtime_ns != recorded_stat['mtime_ns']:
                return None, 'restore_failed_integrity'
        else:
            # Upgrade old snapshots once, preserving expensive prefill work.
            # Subsequent restores and resident hits use the persisted stat record.
            if row['digest'].startswith('text:') or sha256(path) != row['digest']:
                return None, 'restore_failed_integrity'
            row['metadata']['snapshot_stat'] = {'mtime_ns':current_stat.st_mtime_ns}
            with self.db:
                self.db.execute('UPDATE sessions SET metadata=? WHERE id=?',
                    (json.dumps(row['metadata'],ensure_ascii=False),sid))
        return row, 'snapshot_verified'
