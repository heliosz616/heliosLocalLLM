"""Expire native SSD snapshots using metadata mtime (native last-use time).

Call under the gateway's backend lock, or with the inference service stopped.
No payload hashing/decompression is needed; active GPU state is untouched.
"""
from pathlib import Path
import json
import math
import os
import re
import time
import fcntl
from functools import wraps


def quota_locked(function):
    @wraps(function)
    def run(*args, lock_path=None, **kwargs):
        if lock_path is None:
            return function(*args, **kwargs)
        path = Path(lock_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return function(*args, **kwargs)
        finally:
            os.close(fd)
    return run

NAME = re.compile(r'[0-9]+-[0-9]+')
IDENTITY = re.compile(r'[0-9a-f]{64}')


def cache_roots(state, project):
    state, project = Path(state), Path(project)
    roots = [state / name for name in ('q5-ssd-cache', 'flash-next-ssd-cache')]
    return roots


def open_files():
    result = set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    result.add(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            pass
    return result


@quota_locked
def prune_snapshots(roots, retention_seconds=86400, *, now=None, opened=None, clear=False):
    """Remove expired committed pairs and stale orphan payloads, never history.

    Recent metadata protects an old payload: restore touches metadata, not the
    immutable binary. Compression times likewise must not extend retention.
    """
    if not math.isfinite(retention_seconds) or retention_seconds <= 0:
        raise ValueError('KV retention must be positive and finite')
    now = time.time() if now is None else now
    cutoff = now - retention_seconds
    opened = open_files() if opened is None else opened
    report = dict(checked_unix=now, retention_seconds=retention_seconds, explicit_clear=clear,
                  deleted_files=[], removed_allocated_bytes=0, retained_snapshots=0,
                  skipped_open=0, errors=[])
    seen = set()

    def remove(paths):
        if any(str(p.absolute()) in opened for p in paths):
            report['skipped_open'] += 1
            return
        # Metadata first: a partial deletion can only become a cold miss.
        for path in paths:
            if path.is_symlink():
                continue
            try:
                stat = path.stat()
                path.unlink()
                report['deleted_files'].append(str(path))
                report['removed_allocated_bytes'] += stat.st_blocks * 512
            except FileNotFoundError:
                pass
            except OSError as error:
                report['errors'].append(f'{path}: {error}')

    for root in map(Path, roots):
        if root.is_symlink() or not root.exists():
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            dirs[:] = [name for name in dirs if not (parent / name).is_symlink()]
            if not IDENTITY.fullmatch(parent.name):
                continue
            for name in files:
                meta = parent / name
                if meta.suffix != '.json' or not NAME.fullmatch(meta.stem) or meta.is_symlink():
                    continue
                if meta.with_suffix('.reserve').exists():
                    continue
                if str(meta.absolute()) in seen:
                    continue
                seen.add(str(meta.absolute()))
                try:
                    stat = meta.stat()
                    if not clear and stat.st_mtime > cutoff:
                        report['retained_snapshots'] += 1
                        continue
                    if stat.st_size > 32 * 1024**2:
                        continue
                    value = json.loads(meta.read_text())
                    if not isinstance(value, dict) or value.get('version') != 1 or value.get('identity') != parent.name:
                        continue
                    remove([meta, meta.with_suffix('.bin'), meta.with_suffix('.bin.zst')])
                except (OSError, ValueError, TypeError) as error:
                    report['errors'].append(f'{meta}: {error}')
            # A crash/old archive can leave an unindexed payload. Never remove
            # any part of a recent, committed or currently open snapshot.
            for name in files:
                match = re.fullmatch(r'([0-9]+-[0-9]+)\.(bin(?:\.zst)?|part|json\.part)', name)
                if not match:
                    continue
                path = parent / name
                if path.is_symlink() or (parent / (match[1] + '.json')).exists() or (parent / (match[1] + '.reserve')).exists():
                    continue
                try:
                    if clear or path.stat().st_mtime <= cutoff:
                        remove([path])
                except FileNotFoundError:
                    pass
    return report


def cache_bytes(roots, exclude=None):
    """Logical disk bytes, including compressed copies and metadata, once each."""
    excluded = Path(exclude).resolve() if exclude else None
    seen, total = set(), 0
    for root in map(Path, roots):
        if root.is_symlink() or not root.exists():
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            dirs[:] = [n for n in dirs if not (parent / n).is_symlink()]
            if not IDENTITY.fullmatch(parent.name) or parent.resolve() == excluded:
                continue
            for name in files:
                p = parent / name
                if p.is_symlink() or p.resolve() in seen:
                    continue
                seen.add(p.resolve())
                try:
                    total += p.stat().st_size
                except FileNotFoundError:
                    pass
    return total


@quota_locked
def enforce_capacity(roots, maximum_bytes, *, opened=None):
    """Evict oldest last-used committed snapshots across all namespaces."""
    if maximum_bytes <= 0:
        raise ValueError('KV capacity must be positive')
    roots = list(roots)
    opened = open_files() if opened is None else opened
    before = total = cache_bytes(roots)
    candidates = {}
    for root in map(Path, roots):
        if root.is_symlink() or not root.exists():
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            dirs[:] = [n for n in dirs if not (parent / n).is_symlink()]
            if not IDENTITY.fullmatch(parent.name):
                continue
            for name in files:
                meta = parent / name
                if meta.suffix != '.json' or not NAME.fullmatch(meta.stem) or meta.is_symlink():
                    continue
                if meta.with_suffix('.reserve').exists():
                    continue
                try:
                    if meta.stat().st_size > 32 * 1024**2:
                        continue
                    value = json.loads(meta.read_text())
                    if value.get('version') == 1 and value.get('identity') == parent.name:
                        candidates[meta.resolve()] = meta.stat().st_mtime
                except (OSError, ValueError, AttributeError):
                    continue
    removed = []
    for meta in sorted(candidates, key=candidates.get):
        if total <= maximum_bytes:
            break
        paths = [meta, meta.with_suffix('.bin'), meta.with_suffix('.bin.zst')]
        if any(str(p) in opened for p in paths):
            continue
        for p in paths:
            if p.is_symlink():
                continue
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
                removed.append(str(p))
            except FileNotFoundError:
                pass
    return dict(maximum_bytes=maximum_bytes, before_bytes=before,
                after_bytes=cache_bytes(roots), deleted_files=removed)
