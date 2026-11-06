"""Local-only key administration. API clients cannot grant themselves access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['grant','revoke','list'])
    p.add_argument('username',nargs='?')
    args=p.parse_args()
    registry=ROOT/'state'/'users.json'
    registry.parent.mkdir(exist_ok=True)
    users=json.loads(registry.read_text()) if registry.exists() else {}
    if args.action=='list':
        print(json.dumps({k:{'enabled':v['enabled']} for k,v in users.items()},indent=2))
        return
    if not args.username or not re.fullmatch('[A-Za-z0-9_-]{1,40}',args.username) or args.username=='owner':
        raise SystemExit('Supply a username (letters/digits/_/-), excluding owner')
    if args.action=='grant':
        if args.username in users and users[args.username]['enabled']:
            raise SystemExit('User already enabled; revoke first to rotate the key')
        key=secrets.token_urlsafe(36)
        users[args.username]={'enabled':True,'sha256':hashlib.sha256(key.encode()).hexdigest()}
    else:
        if args.username not in users: raise SystemExit('Unknown user')
        users[args.username]['enabled']=False
    temp=registry.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(users,f,indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp,registry)
    if args.action=='grant':
        print('Username:',args.username)
        print('API key (shown once):',key)
    else:
        print('Revoked:',args.username)

if __name__=='__main__': main()
