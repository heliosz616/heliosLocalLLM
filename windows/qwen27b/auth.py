"""Local principal registry: only token digests are retained for delegated users."""
import hashlib
import hmac
import json
from pathlib import Path

def principal(token, owner_key, registry):
    if hmac.compare_digest(token.encode(),owner_key.encode()):
        return 'owner'
    path=Path(registry)
    if not path.exists():
        return None
    try:
        users=json.loads(path.read_text())
    except (OSError,ValueError):
        return None
    digest=hashlib.sha256(token.encode()).hexdigest()
    for name,entry in users.items():
        if entry.get('enabled') and hmac.compare_digest(digest,entry['sha256']):
            return name
    return None

def session_key(who,sid):
    if sid is None or who=='owner':
        return sid
    return hashlib.sha256(who.encode()).hexdigest()+':'+sid
