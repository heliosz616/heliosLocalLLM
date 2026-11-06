"""Bound each activated native cache by available disk space shared by all models."""
from pathlib import Path
import json
import shutil
import time

GIB = 1024**3


def committed_bytes(namespace, identity):
    """Mirror native startup admission; never count an orphan payload as evictable."""
    total = 0
    for metadata_path in Path(namespace).glob('*.json'):
        try:
            name = metadata_path.stem
            if not name or any(char not in '0123456789-' for char in name):
                continue
            if metadata_path.stat().st_size > 32 * 1024**2:
                continue
            value = json.loads(metadata_path.read_text())
            if value.get('version') != 1 or value.get('identity') != identity:
                continue
            size = value['bytes']
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                continue
            if not isinstance(value.get('checksum'), str) or len(value['checksum']) != 32:
                continue
            payload = metadata_path.with_suffix('.bin')
            if payload.is_file() and payload.stat().st_size == size:
                total += size
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return total


def select_budget(namespace, identity, requested_bytes, reserve_bytes=32 * GIB, *, free_bytes=None):
    """Reserve free space at activation without shrinking existing cache for that reason.

    The service runs one physical backend. Each subsequent model activation
    recalculates its allowance from the free space left by the previous one.
    An explicitly configured maximum still controls native LRU pruning.
    """
    namespace = Path(namespace)
    if len(identity) != 64 or any(char not in '0123456789abcdef' for char in identity):
        raise ValueError('Invalid native cache identity')
    if requested_bytes <= 0 or reserve_bytes < 0:
        raise ValueError('Cache maximum must be positive and disk reserve nonnegative')
    existing = committed_bytes(namespace, identity)
    free = shutil.disk_usage(namespace).free if free_bytes is None else free_bytes
    growth = max(0, free - reserve_bytes)
    limit = max(1, min(requested_bytes, existing + growth))
    return dict(namespace=str(namespace), existing_committed_bytes=existing,
                requested_maximum_bytes=requested_bytes, effective_maximum_bytes=limit,
                free_at_activation_bytes=free, disk_reserve_bytes=reserve_bytes,
                maximum_new_payload_growth_bytes=max(0, limit - existing), checked_unix=time.time(),
                scope='Activation-time bound on service cache growth; native LRU still applies. Pending snapshots need temporary space. Other applications can independently consume disk.')


def cap_native_cache(environment, reserve_bytes=32 * GIB, *, roots=None, total_maximum_bytes=None):
    root = environment.get('LLAMA_CACHE_SSD_DIR')
    identity = environment.get('LLAMA_CACHE_SSD_ID')
    maximum = environment.get('LLAMA_CACHE_SSD_BYTES')
    if not all((root, identity, maximum)):
        return None
    capacity = None
    if roots is not None and total_maximum_bytes is not None:
        from cache_retention import enforce_capacity, cache_bytes
        capacity = enforce_capacity(roots, total_maximum_bytes, lock_path=Path(root).parent / '.kv-quota.lock')
        other_bytes = cache_bytes(roots, exclude=Path(root) / identity)
        if environment.get('LLAMA_RECURRENT_CHECKPOINTS') == '1':
            # Both native writers reserve globally before creating any KV or
            # recurrent payload. In-flight bytes participate in that quota.
            environment.update(
                LLAMA_CACHE_SSD_SHARED_ROOTS=json.dumps([str(Path(p).resolve()) for p in roots]),
                LLAMA_CACHE_SSD_SHARED_LOCK=str(Path(root).parent / '.kv-quota.lock'),
                LLAMA_CACHE_SSD_SHARED_BYTES=str(total_maximum_bytes))
            maximum = min(int(maximum), total_maximum_bytes)
        else:
            # Legacy writers reserve each activated model's remaining share.
            maximum = min(int(maximum), max(1, total_maximum_bytes - other_bytes - 1024**2))
    report = select_budget(Path(root) / identity, identity, int(maximum), reserve_bytes)
    if capacity is not None:
        report.update(total_maximum_bytes=total_maximum_bytes,
                      other_namespaces_bytes=other_bytes, capacity_cleanup=capacity)
    environment['LLAMA_CACHE_SSD_BYTES'] = str(report['effective_maximum_bytes'])
    return report
