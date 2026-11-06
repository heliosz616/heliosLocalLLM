"""Real streaming A/B/A and post-restart cache test, including client-observed TTFT."""
import argparse
import json
import time
import uuid
from pathlib import Path
import httpx
from openai import OpenAI
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['create','resume'])
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.local.json').read_text())
    key=(ROOT/'state'/'api-key.txt').read_text().strip()
    base=f'http://{cfg["host"]}:{cfg["port"]}'
    http=httpx.Client(trust_env=False,timeout=3600,headers={'Authorization':'Bearer '+key})
    client=OpenAI(base_url=base+'/v1',api_key=key,http_client=http,timeout=3600,max_retries=0)
    statepath=ROOT/'state'/'stream-regression.json'
    output=ROOT/'reports'/f'cache-stream-{args.mode}-{int(time.time())}.json'
    report={'mode':args.mode,'results':[]}
    def request(label,sid,messages):
        start=time.perf_counter();first=None;text='';usage=None;timings=None
        with client.chat.completions.create(model=cfg['model'],messages=messages,temperature=0,seed=42,max_tokens=48,
            stream=True,stream_options={'include_usage':True},extra_body={'reasoning_effort':'none','session_id':sid}) as stream:
            for chunk in stream:
                data=chunk.model_dump()
                if chunk.usage:usage=data['usage']
                if data.get('timings'):timings=data['timings']
                for choice in chunk.choices:
                    if choice.delta.content:
                        if first is None:first=time.perf_counter()-start
                        text+=choice.delta.content
        elapsed=time.perf_counter()-start
        response=http.get(base+'/sessions/'+sid);response.raise_for_status()
        metrics=response.json()['last_request']
        result=dict(label=label,text=text,ttft_seconds=first,end_to_end_seconds=elapsed,
                    usage=usage,timings=timings,cache=metrics)
        report['results'].append(result)
        output.write_text(json.dumps(report,indent=2))
        print(json.dumps(result),flush=True)
        return result
    if args.mode=='create':
        suffix=uuid.uuid4().hex[:8]
        messages=[{'role':'system','content':'Remember the marker. Answer only with the marker when asked.'},
            {'role':'user','content':'The marker is APRICOT-731.\n'+('The orchard contains apples, pears, and quiet walking paths. '*750)+'\nWhat is the marker?'}]
        bmessages=[{'role':'user','content':'The marker is COBALT-952. Reply only with this marker.'}]
        state=dict(A='stream-A-'+suffix,B='stream-B-'+suffix,messages=messages,bmessages=bmessages)
        cold=request('A_cold',state['A'],messages)
        assert cold['cache']['prompt_tokens']>=8000
        same=request('A_same',state['A'],messages)
        assert same['cache']['hit_ratio']>.98
        assert same['cache']['restore_status']=='resident_prefix_hit'
        assert same['text']==cold['text']
        b=request('B_cold',state['B'],bmessages)
        assert 'COBALT' in b['text']
        state['continuation']=messages+[{'role':'assistant','content':same['text']},{'role':'user','content':'Repeat my marker exactly.'}]
        back=request('A_after_B',state['A'],state['continuation'])
        assert back['cache']['hit_ratio']>.98
        assert 'APRICOT' in back['text'] and 'COBALT' not in back['text']
        state['expected']=back['text']
        statepath.write_text(json.dumps(state))
    else:
        state=json.loads(statepath.read_text())
        a=request('A_after_restart',state['A'],state['continuation'])
        assert a['cache']['restore_status']=='prefix_hit' and a['cache']['hit_ratio']>.98
        assert a['text']==state['expected']
        hot=request('A_hot_after_restart',state['A'],state['continuation'])
        assert hot['cache']['restore_status']=='resident_prefix_hit' and hot['cache']['hit_ratio']>.98
        assert hot['text']==state['expected']
        b=request('B_after_restart',state['B'],state['bmessages'])
        assert 'COBALT' in b['text']
    report['passed']=True
    output.write_text(json.dumps(report,indent=2))
    print(output)

if __name__=='__main__':main()
