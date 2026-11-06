"""Administrator-defined model allowlist; request model names never become paths."""
import json
from pathlib import Path
import time


def ensure_catalog(root, base):
    """Enable supervisor recovery for fresh and existing single-model installs."""
    path = Path(root) / 'models.local.json'
    if not path.exists():
        # Preserve the legacy single-model manifest and snapshot location.
        atomic_json(path, {'models': load_catalog(root, base)})
    load_catalog(root, base, path.name)
    return path.name


def load_catalog(root, base, filename=None):
    if not filename:
        return {base['model']: dict(config=base.copy(), manifest='manifest.json', slots='slots')}
    data = json.loads((Path(root)/filename).read_text(encoding='utf-8'))
    profiles = data['models']
    if not profiles or base['model'] not in profiles:
        raise ValueError('Default model must be registered')
    dirs = set()
    for name, profile in profiles.items():
        for field in ('manifest', 'slots'):
            value = profile[field]
            if not isinstance(value, str) or value in ('', '.', '..') or Path(value).name != value or '/' in value or '\\' in value:
                raise ValueError('Model profile paths must be plain names')
        if profile['slots'] in dirs:
            raise ValueError('Models must have separate snapshot directories')
        dirs.add(profile['slots'])
        cfg = profile['config']
        for field in ('runtime_directory','runtime_manifest'):
            value=cfg.get(field)
            if value is not None and (not isinstance(value,str) or value in ('','.','..') or Path(value).name!=value or '/' in value or '\\' in value):
                raise ValueError('Runtime paths must be plain names')
        if cfg['model'] != name:
            raise ValueError('Model alias mismatch')
        for field in ('host', 'port', 'upstream', 'allowed_networks'):
            if cfg[field] != base[field]:
                raise ValueError('Models must share gateway network policy')
    return profiles


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    # Windows readers/antivirus can briefly hold a destination without delete sharing.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


def prune_shared(stores, limit, incoming, protect):
    """One SSD quota across model stores; caller owns the global generation lock."""
    rows=[]
    for name, store in stores.items():
        rows.extend((accessed,name,sid,size) for sid,size,accessed in store.entries())
    total=sum(row[3] for row in rows)
    for accessed,name,sid,size in sorted(rows):
        if (name,sid)==protect:
            continue
        target=stores[name]
        if total+incoming>limit or time.time()-accessed>target.config['session_ttl_seconds']:
            target.delete(sid)
            total-=size
    if total+incoming>limit:
        raise ValueError('Shared snapshot quota exceeded; previous snapshot preserved')
