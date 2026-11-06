"""Real hot-session growth: render full chat history, append ~16K, keep cumulative context."""
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
    parser.add_argument('--chunk-tokens',type=int,default=16384)
    parser.add_argument('--generate',type=int,default=128)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.local.json').read_text())
    assert cfg['context']==147456,'Run the 144K configuration for this workload'
    assert args.chunk_tokens in (16384,32768)
    session='incremental-'+uuid.uuid4().hex[:10]
    output=ROOT/'reports'/f'incremental-{args.chunk_tokens}-{int(time.time())}.json'
    internal=(ROOT/'state/internal-key.txt').read_text().strip()
    key=(ROOT/'state/api-key.txt').read_text().strip()
    native=httpx.Client(base_url=cfg['upstream'],trust_env=False,timeout=120,
        headers={'Authorization':'Bearer '+internal})
    http=httpx.Client(trust_env=False,timeout=3600)
    client=OpenAI(base_url=f'http://{cfg["host"]}:{cfg["port"]}/v1',api_key=key,http_client=http,
        max_retries=0,timeout=3600)
    def tokenize(text):
        response=native.post('/tokenize',json={'content':text,'add_special':False,'parse_special':True})
        response.raise_for_status()
        return response.json()['tokens']
    seed=tokenize(' The journal records quiet rivers, trees, mathematics and science.')
    messages=[{'role':'system','content':'The permanent archive code is RIVER-7492. When asked, reply only with that code.'}]
    results=[]
    previous_prompt=0
    target=0
    while target<cfg['context']-args.generate:
        target=min(target+args.chunk_tokens,cfg['context']-args.generate)
        pending={'role':'user','content':'Additional journal entries:\n__INCREMENTAL_FILLER__\nWhat is the permanent archive code?'}
        response=native.post('/apply-template',json={'messages':messages+[pending],
            'chat_template_kwargs':{'enable_thinking':False}})
        response.raise_for_status()
        before,after=response.json()['prompt'].split('__INCREMENTAL_FILLER__')
        prefix,suffix=tokenize(before),tokenize(after)
        budget=target-len(prefix)-len(suffix)
        assert budget>0,'Chunk is too small for the added history'
        user_template=pending['content']
        # Canonically tokenize the WHOLE rendered prompt. Concatenating separately
        # tokenized pieces can change a BPE boundary near the beginning of history
        # and accidentally turn the next round into a full cache miss.
        for _ in range(12):
            filler=(seed*((budget+len(seed)-1)//len(seed)))[:budget]
            response=native.post('/detokenize',json={'tokens':filler})
            response.raise_for_status()
            pending['content']=user_template.replace('__INCREMENTAL_FILLER__',response.json()['content'])
            response=native.post('/apply-template',json={'messages':messages+[pending],
                'chat_template_kwargs':{'enable_thinking':False}})
            response.raise_for_status()
            tokens=tokenize(response.json()['prompt'])
            if len(tokens)==target:
                break
            budget+=target-len(tokens)
            assert budget>0
        assert target-8<=len(tokens)<=target,'Unable to fit cumulative prompt safely'
        started=time.time();start=time.perf_counter();first=None;text='';timings={};usage={}
        with client.completions.create(model=cfg['model'],prompt=tokens,temperature=0,seed=42,
                max_tokens=args.generate,stream=True,stream_options={'include_usage':True},
                extra_body={'reasoning_effort':'none','session_id':session,'ignore_eos':True}) as stream:
            for chunk in stream:
                data=chunk.model_dump()
                if data.get('timings'):timings.update(data['timings'])
                if data.get('usage'):usage.update(data['usage'])
                for choice in chunk.choices:
                    if choice.text:
                        if first is None:first=time.perf_counter()-start
                        text+=choice.text
        elapsed=time.perf_counter()-start
        response=http.get(f'http://{cfg["host"]}:{cfg["port"]}/sessions/{session}',
            headers={'Authorization':'Bearer '+key})
        response.raise_for_status()
        metrics=response.json()['last_request']
        assert usage['prompt_tokens']==len(tokens)
        assert 'RIVER-7492' in text
        if previous_prompt:
            assert metrics['restore_status']=='resident_prefix_hit',metrics
            # Retokenization of generated text can rewind a small checkpoint suffix.
            assert metrics['cached_tokens']>=previous_prompt-1024,metrics
        result=dict(round=len(results)+1,started_unix=started,finished_unix=time.time(),
            cumulative_prompt_tokens=len(tokens),target_prompt_tokens=target,requested_increment=len(tokens)-previous_prompt,
            actual_new_prefill_tokens=metrics['prefill_tokens'],cached_tokens=metrics['cached_tokens'],
            ttft_seconds=first,end_to_end_seconds=elapsed,timings=timings,cache=metrics,needle_passed=True)
        results.append(result)
        output.write_text(json.dumps(dict(config=cfg,session=session,chunk_tokens=args.chunk_tokens,
            forced_ignore_eos=True,results=results,passed=False),indent=2))
        print(json.dumps(result),flush=True)
        messages.extend([pending,{'role':'assistant','content':text}])
        previous_prompt=len(tokens)
    report=json.loads(output.read_text());report['passed']=True
    output.write_text(json.dumps(report,indent=2))
    print(str(output),flush=True)
    client.close();native.close()


if __name__=='__main__':
    main()
