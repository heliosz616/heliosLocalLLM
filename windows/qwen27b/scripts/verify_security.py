"""HTTP access checks against the running gateway; no model inference needed."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import uuid
import httpx

ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'config.local.json').read_text())
    base=f'http://{cfg["host"]}:{cfg["port"]}'
    registry=ROOT/'state'/'users.json'
    users=json.loads(registry.read_text()) if registry.exists() else {}
    names=['test-'+uuid.uuid4().hex for _ in range(2)]
    keys=[secrets.token_urlsafe(36) for _ in names]
    report={}
    def write():
        tmp=registry.with_suffix('.tmp')
        tmp.write_text(json.dumps(users,indent=2)); os.replace(tmp,registry)
    try:
        for name,key in zip(names,keys):
            users[name]={'enabled':True,'sha256':hashlib.sha256(key.encode()).hexdigest()}
        write()
        with httpx.Client(base_url=base,trust_env=False,timeout=30) as c:
            assert c.get('/v1/models').status_code==401
            assert c.get('/v1/models',headers={'Authorization':'Bearer incorrect'}).status_code==401
            report['missing_and_invalid_key']=401
            ha={'Authorization':'Bearer '+keys[0]}; hb={'Authorization':'Bearer '+keys[1]}
            assert c.get('/v1/models',headers=ha).status_code==200
            assert c.post('/sessions',headers=ha,json={'session_id':'security-check'}).status_code==200
            assert c.get('/sessions/security-check',headers=hb).status_code==404
            assert c.post('/sessions',headers=hb,json={'session_id':'security-check'}).status_code==200
            assert c.delete('/sessions/security-check',headers=hb).status_code==200
            assert c.get('/sessions/security-check',headers=ha).status_code==200
            assert c.delete('/sessions/security-check',headers=ha).status_code==200
            report['cross_user_session_isolation']=True
            assert c.get('/slots',headers=ha).status_code==404
            report['gateway_slots_not_exposed']=True
            users[names[0]]['enabled']=False; write()
            assert c.get('/v1/models',headers=ha).status_code==401
            report['revocation_without_restart']=True
        with httpx.Client(base_url=cfg['upstream'],trust_env=False,timeout=30) as c:
            assert c.get('/slots').status_code==401
            external_key=(ROOT/'state'/'api-key.txt').read_text().strip()
            assert c.get('/slots',headers={'Authorization':'Bearer '+external_key}).status_code==401
            internal=(ROOT/'state'/'internal-key.txt').read_text().strip()
            assert c.get('/slots',headers={'Authorization':'Bearer '+internal}).status_code==200
            report['internal_key_separated']=True
        print(json.dumps(report,indent=2))
        (ROOT/'reports'/'security-http.json').write_text(json.dumps(report,indent=2))
    finally:
        for name in names: users.pop(name,None)
        write()

if __name__=='__main__': main()
