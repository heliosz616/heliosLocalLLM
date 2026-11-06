import json
from pathlib import Path
import time
import httpx
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'config.local.json').read_text())
key=(ROOT/'state'/'api-key.txt').read_text().strip()
with httpx.Client(trust_env=False,timeout=2) as c:
    for _ in range(120):
        try:
            r=c.get(f'http://{cfg["host"]}:{cfg["port"]}/health',headers={'Authorization':'Bearer '+key})
            if r.status_code==200:
                print(r.text)
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    else:
        raise SystemExit('Gateway readiness timeout')
