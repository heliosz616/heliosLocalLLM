#!/usr/bin/env python3
"""Create a reviewed-index ZIP, excluding ignored files and all Git metadata."""
from pathlib import Path
import argparse
import json
import zipfile
from audit_release import ROOT,audit,staged,blobs,git

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--private-values',type=Path)
args=parser.parse_args()
values=json.loads(args.private_values.read_text()) if args.private_values else []
result=audit(private_values=values)
if result['findings']:raise SystemExit('Publication audit failed')
if git('diff','--name-only'):raise SystemExit('Unstaged tracked changes: restage and review before packaging')
entries=staged();data=blobs(oid for _,oid in entries.values())
with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
    for path,(mode,oid) in sorted(entries.items()):
        item=zipfile.ZipInfo('heliosLocalLLM/'+path,date_time=(2026,1,1,0,0,0))
        item.create_system=3
        item.external_attr=(int(mode,8)<<16)
        archive.writestr(item,data[oid],compress_type=zipfile.ZIP_DEFLATED)
print('Source package created from audited Git index; no Git history or filesystem extras included.')
