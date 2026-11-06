"""Real SDK smoke / SSD prefix-reuse checks. Never substitutes mock inference."""
import argparse
import json
from pathlib import Path
import time
import uuid
import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('suite',choices=['smoke','cache','resume'])
    p.add_argument('--base-url')
    p.add_argument('--prompt-tokens',type=int,default=8500)
    args = p.parse_args()
    cfg = json.loads((ROOT/'config.local.json').read_text())
    import os
    key = (ROOT/'state'/'api-key.txt').read_text().strip()
    base = args.base_url or f'http://{cfg["host"]}:{cfg["port"]}/v1'
    client = OpenAI(base_url=base,api_key=key,timeout=3600,max_retries=0,http_client=httpx.Client(trust_env=False))
    report = dict(suite=args.suite,base_url=base,results=[])
    output = ROOT/'reports'/f'{args.suite}-{int(time.time())}.json'
    def record(label,value):
        report['results'].append(dict(label=label,**value))
        output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(label,json.dumps(value,ensure_ascii=False),flush=True)
    def chat(messages,sid=None,**extra):
        start = time.perf_counter()
        r = client.chat.completions.create(model=cfg['model'],messages=messages,temperature=0,seed=42,
            reasoning_effort='none',max_tokens=48,extra_body=dict(session_id=sid,**extra) if sid else extra)
        return r,dict(response=r.model_dump(),elapsed=time.perf_counter()-start)
    if args.suite == 'smoke':
        record('models',dict(models=client.models.list().model_dump()))
        messages=[{'role':'system','content':'Answer briefly and accurately.'},{'role':'user','content':'What is 17 + 25?'}]
        r, data=chat(messages)
        assert '42' in (r.choices[0].message.content or ''),r
        record('chat',data)
        common=client.chat.completions.create(model=cfg['model'],messages=[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],
            temperature=0.3,top_p=0.9,reasoning_effort='none',max_tokens=32)
        assert '4' in (common.choices[0].message.content or '')
        record('common_sampling_fields',dict(response=common.model_dump()))
        thinking=client.chat.completions.create(model=cfg['model'],messages=[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],
            temperature=0,top_p=0.9,reasoning_effort='low',max_tokens=96)
        message=thinking.choices[0].message.model_dump()
        assert message.get('reasoning_content') or message.get('content')
        record('reasoning_effort_low',dict(response=thinking.model_dump()))
        messages += [r.choices[0].message.model_dump(exclude_none=True),{'role':'user','content':'Multiply that result by two.'}]
        r,data=chat(messages)
        assert '84' in (r.choices[0].message.content or ''),r
        record('multiturn',data)
        start=time.perf_counter(); first=None; chunks=[]
        with client.chat.completions.create(model=cfg['model'],messages=[{'role':'user','content':'Count from 1 to 5.'}],max_tokens=32,
            stream=True,stream_options={'include_usage':True},extra_body={'session_id':'smoke-stream-'+uuid.uuid4().hex[:8]}) as stream:
            for chunk in stream:
                if first is None: first=time.perf_counter()-start
                chunks.append(chunk.model_dump())
        assert chunks
        record('stream',dict(ttft=first,elapsed=time.perf_counter()-start,chunks=chunks))
        tools=[{'type':'function','function':{'name':'get_temperature','description':'Look up temperature for a city',
            'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
        msgs=[{'role':'user','content':'Use get_temperature to get the temperature in Example City A.'}]
        r=client.chat.completions.create(model=cfg['model'],messages=msgs,tools=tools,
            tool_choice={'type':'function','function':{'name':'get_temperature'}},temperature=0,reasoning_effort='none',max_tokens=150)
        calls=r.choices[0].message.tool_calls
        assert calls and calls[0].function.name=='get_temperature',r
        record('tool_call',dict(response=r.model_dump()))
        msgs += [r.choices[0].message.model_dump(exclude_none=True),{'role':'tool','tool_call_id':calls[0].id,'content':'{"temperature_celsius":23}'}]
        r=client.chat.completions.create(model=cfg['model'],messages=msgs,tools=tools,temperature=0,reasoning_effort='none',max_tokens=64)
        assert '23' in (r.choices[0].message.content or ''),r
        record('tool_result',dict(response=r.model_dump()))
        r=client.completions.create(model=cfg['model'],prompt='The capital of France is',temperature=0,max_tokens=12,stop=['\n'])
        record('completion',dict(response=r.model_dump()))
        http=httpx.Client(trust_env=False)
        assert http.get(base.removesuffix('/v1')+'/v1/models').status_code==401
        assert http.get(base.removesuffix('/v1')+'/slots',headers={'Authorization':'Bearer '+key}).status_code==404
        record('access_control',dict(unauthenticated=401,slots_not_exposed=404))
    elif args.suite=='cache':
        # Synthetic public test text only. Exact length is measured with native tokenizer.
        upstream=httpx.Client(base_url=cfg['upstream'],timeout=120,trust_env=False,
            headers={'Authorization':'Bearer '+(ROOT/'state'/'internal-key.txt').read_text().strip()})
        filler='The orchard contains apples, pears, and quiet walking paths. '
        text=filler*1200
        tokens=upstream.post('/tokenize',json={'content':text,'add_special':False}).json()['tokens'][:args.prompt_tokens]
        text=upstream.post('/detokenize',json={'tokens':tokens}).json()['content']
        suffix=uuid.uuid4().hex[:8]
        a='regression-A-'+suffix; b='regression-B-'+suffix
        msgs=[{'role':'system','content':'Remember the private marker for this conversation. Reply with the marker only when asked.'},
              {'role':'user','content':'The marker is APRICOT-731. Background notes:\n'+text+'\nWhat is the marker?'}]
        r,data=chat(msgs,a); record('A_cold',data)
        assert data['response']['session_cache']['prompt_tokens']>=args.prompt_tokens
        r,data=chat(msgs,a); record('A_same',data)
        assert data['response']['session_cache']['hit_ratio']>0.8
        bmsgs=[{'role':'user','content':'Your marker is COBALT-952. Repeat this marker.'}]
        r,data=chat(bmsgs,b); record('B_cold',data)
        continuation=msgs+[{'role':'assistant','content':'APRICOT-731'},{'role':'user','content':'Repeat my marker exactly.'}]
        r,data=chat(continuation,a); record('A_after_B',data)
        assert 'APRICOT' in (r.choices[0].message.content or '') and 'COBALT' not in (r.choices[0].message.content or '')
        assert data['response']['session_cache']['hit_ratio']>0.8
        expected=r.choices[0].message.content
        state=dict(A=a,B=b,messages=continuation,bmessages=bmsgs,expected=expected)
        (ROOT/'state'/'regression.json').write_text(json.dumps(state))
        record('restart_pending',dict(instruction='Restart service, then run validate.py resume'))
    else:
        state=json.loads((ROOT/'state'/'regression.json').read_text())
        r,data=chat(state['messages'],state['A']); record('A_after_restart',data)
        assert data['response']['session_cache']['restore_status']=='prefix_hit'
        assert data['response']['session_cache']['hit_ratio']>0.8
        assert r.choices[0].message.content==state['expected'],'Deterministic output differs'
        r,data=chat(state['bmessages'],state['B']); record('B_after_restart',data)
        assert 'COBALT' in (r.choices[0].message.content or '')
    report['passed']=True
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(output)

if __name__=='__main__':
    main()
