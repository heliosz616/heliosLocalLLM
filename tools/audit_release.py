#!/usr/bin/env python3
"""Audit the Git index and reachable history without printing matched secrets."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private-key': rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----',
    'provider-token': rb'\b(?:gh[pousr]_|github_pat_|hf_|sk-(?:proj-|live-)?)\w[\w-]{20,}',
    'personal-home': rb'(?:/home/|/Users/)[a-zA-Z0-9_.-]+(?:/|\b)|[A-Za-z]:[\\/]Users[\\/][^\\/\s"\x27]+',
    'private-address': rb'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b',
    'overlay-hostname': rb'\b[\w.-]+\.ts\.net\b',
    'email': rb'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',
    'mac-address': rb'\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b',
}
COMPILED = {name: re.compile(pattern) for name, pattern in RULES.items()}
BINARY_SUFFIXES = {'.gguf','.safetensors','.onnx','.bin','.so','.dll','.exe','.o','.a','.hsaco','.sqlite','.sqlite-wal','.sqlite-shm','.db','.log','.jsonl','.pyc','.zip','.gz','.png','.jpg','.jpeg','.wav','.mp3','.pdf'}
PRIVATE_PREFIXES = ('linux/state/','linux/runtime/','linux/models/','linux/reports/',
                    'windows/qwen27b/state/','windows/qwen27b/models/','windows/qwen27b/reports/',
                    'windows/qwen27b/build/','windows/qwen27b/vendor/')


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def blobs(ids):
    ids = sorted(set(ids))
    if not ids: return {}
    data = git('cat-file','--batch') if not ids else subprocess.check_output(
        ['git','-C',str(ROOT),'cat-file','--batch'], input=('\n'.join(ids)+'\n').encode())
    result = {}; offset = 0
    for oid in ids:
        end = data.index(b'\n',offset); header = data[offset:end].split(); size = int(header[2]); offset=end+1
        if header[1] == b'blob': result[oid] = data[offset:offset+size]
        offset += size+1
    return result


def staged():
    result={}
    for entry in git('ls-files','--stage','-z').split(b'\0'):
        if not entry: continue
        meta,path=entry.split(b'\t',1);mode,oid,stage=meta.decode().split()
        if stage!='0': raise RuntimeError('Unmerged index')
        result[path.decode()] = (mode,oid)
    return result


def inspect(path, data, allow, private_values):
    findings=[]; parts=Path(path).parts
    lowered = data.lower()
    path_bytes = path.encode()
    if path.startswith(PRIVATE_PREFIXES) or any(p in ('.git','__pycache__','.pytest_cache') or p.startswith('.venv') for p in parts) or '.local.' in Path(path).name or Path(path).name.startswith('.env') or Path(path).name in ('api-key','api-keys.json','artifact-seal.json'):
        findings.append((path,0,'private-runtime-path'))
    if Path(path).suffix.lower() in BINARY_SUFFIXES or b'\0' in data:
        findings.append((path,0,'binary-or-runtime-payload'))
    for value in private_values:
        encoded=value.encode()
        if encoded and ((encoded.lower() in lowered or encoded.lower() in path_bytes.lower()) if len(encoded)>=8 else re.search(rb'\b'+re.escape(encoded)+rb'\b',data+b'\n'+path_bytes,re.I)):
            findings.append((path,0,'known-private-value')); break
    for name,pattern in COMPILED.items():
        if pattern.search(path_bytes): findings.append((path,0,name))
    for number,line in enumerate(data.splitlines(),1):
        for name,pattern in COMPILED.items():
            if pattern.search(line):
                key=(path,name,hashlib.sha256(line).hexdigest())
                if key not in allow: findings.append((path,number,name))
    return findings


def audit(history=True, private_values=()):
    exceptions=json.loads((ROOT/'tools/privacy_allowlist.json').read_text())
    allow={(x['path'],x['rule'],x['line_sha256']) for x in exceptions}
    entries=staged(); content=blobs(oid for _,oid in entries.values()); findings=[]
    for path,(mode,oid) in entries.items():
        if mode not in ('100644','100755'):findings.append((path,0,'unsupported-git-mode'))
        findings.extend(inspect(path,content[oid],allow,private_values))
    history_count=0
    if history:
        names={}
        for line in git('rev-list','--objects','--all').decode().splitlines():
            parts=line.split(' ',1)
            if len(parts)==2:names[parts[0]]=parts[1]
        old=blobs(names)
        for oid,data in old.items():
            history_count+=1
            findings.extend(inspect(names[oid],data,allow,private_values))
        metadata=git('log','--all','--format=%an%n%ae%n%cn%n%ce%n%B')
        for value in private_values:
            if value and ((value.encode().lower() in metadata.lower()) if len(value)>=8 else re.search(rb'\b'+re.escape(value.encode())+rb'\b',metadata,re.I)):
                findings.append(('git-history',0,'known-private-metadata'))
        for name,pattern in COMPILED.items():
            if name != 'email' and pattern.search(metadata):
                findings.append(('git-history',0,name))
        for address in re.findall(COMPILED['email'],metadata):
            if not address.endswith(b'@users.noreply.github.com'):
                findings.append(('git-history',0,'non-noreply-author-email'))
    findings=sorted(set(findings))
    for path,line,rule in findings:print(f'{path}:{line}: {rule}')
    print(f'Audited {len(entries)} staged files and {history_count} reachable historical blobs; findings: {len(findings)}')
    return {'staged_files':len(entries),'historical_blobs':history_count,'findings':len(findings),'index_tree':git('write-tree').decode().strip()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-values',type=Path,help='Local-only JSON array of known secrets/identifiers; never committed')
    parser.add_argument('--no-history',action='store_true')
    args=parser.parse_args()
    values=json.loads(args.private_values.read_text()) if args.private_values else []
    result=audit(not args.no_history,values)
    raise SystemExit(bool(result['findings']))
