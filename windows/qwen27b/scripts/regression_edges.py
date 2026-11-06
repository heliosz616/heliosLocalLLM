"""Hardware-backed cancellation, corruption and edited-prefix regression checks."""
import concurrent.futures
import json
from pathlib import Path
import sqlite3
import time
import uuid
import httpx
from openai import OpenAI

ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'config.local.json').read_text())
    key=(ROOT/'state'/'api-key.txt').read_text().strip()
    base=f'http://{cfg["host"]}:{cfg["port"]}'
    client=OpenAI(base_url=base+'/v1',api_key=key,timeout=3600,max_retries=0,http_client=httpx.Client(trust_env=False))
    suffix=uuid.uuid4().hex[:8]
    sid='edge-'+suffix
    report={}
    def chat(msgs,session=sid):
        return client.chat.completions.create(model=cfg['model'],messages=msgs,max_tokens=24,temperature=0,seed=42,extra_body={'reasoning_effort':'none','session_id':session})
    messages=[{'role':'user','content':'Remember my unique code LIME-483. Reply with only that code.'}]
    first=chat(messages)
    assert 'LIME' in (first.choices[0].message.content or '')
    catalog=json.loads((ROOT/'models.local.json').read_text()) if (ROOT/'models.local.json').exists() else None
    slots=ROOT/'state'/(catalog['models'][cfg['model']]['slots'] if catalog else 'slots')
    db=sqlite3.connect(slots/'sessions.sqlite')
    old=db.execute('SELECT filename,digest FROM sessions WHERE id=?',(sid,)).fetchone()
    stream=client.chat.completions.create(model=cfg['model'],messages=messages+[first.choices[0].message.model_dump(exclude_none=True),
        {'role':'user','content':'Explain a synthetic data-processing pipeline in at least 1000 words.'}],max_tokens=1024,stream=True,extra_body={'reasoning_effort':'none','session_id':sid})
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            break
    stream.close()
    # A subsequent generation cannot begin until cancellation cleanup has completed.
    started=time.perf_counter()
    r=chat([{'role':'user','content':'What is 2 + 2? Reply only with the number.'}],session='edge-other-'+suffix)
    assert '4' in (r.choices[0].message.content or '')
    assert db.execute('SELECT filename,digest FROM sessions WHERE id=?',(sid,)).fetchone()==old
    report['cancelled_stream_preserved_previous_snapshot']=True
    report['next_request_after_cancel_seconds']=time.perf_counter()-started
    # Corrupt only this regression's own snapshot; gateway must detect before native restore.
    path=slots/old[0]
    with path.open('r+b') as f:
        f.seek(-1,2); b=f.read(1); f.seek(-1,2); f.write(bytes([b[0]^255])); f.flush()
    r=chat(messages)
    assert r.model_extra['session_cache']['restore_status']=='restore_failed_integrity'
    assert r.model_extra['session_cache']['cached_tokens']==0
    report['corrupt_snapshot_rejected']=True
    db.execute('UPDATE sessions SET version=? WHERE id=?',('test-incompatible',sid)); db.commit()
    r=chat(messages)
    assert r.model_extra['session_cache']['restore_status']=='version_mismatch'
    report['version_mismatch_rejected']=True
    changed=[{'role':'system','content':'Respond only with the exact code requested.'},
             {'role':'user','content':'My code is PEAR-619. Reply only with my code.'}]
    r=chat(changed)
    assert 'PEAR' in (r.choices[0].message.content or '')
    assert r.model_extra['session_cache']['cached_tokens']<10
    report['edited_system_prefix_safe']=True
    def concurrent_request(i):
        code=f'PLUM-{i}79'
        r=chat([{'role':'user','content':f'Reply with exactly {code}.'}],session=f'concurrent-{suffix}-{i}')
        assert code in (r.choices[0].message.content or '')
        return r.model_extra['session_cache']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        report['concurrent_sessions']=list(pool.map(concurrent_request,range(3)))
    report['passed']=True
    (ROOT/'reports'/'regression-edges.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    db.close()

if __name__=='__main__': main()
