"""Measured exact-length native prompts through authenticated gateway."""
import argparse
import json
import os
from pathlib import Path
import time
import httpx
from openai import OpenAI

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prompt-tokens',type=int,required=True)
    p.add_argument('--generate',type=int,default=128)
    p.add_argument('--session',default=None)
    p.add_argument('--needle',action='store_true')
    p.add_argument('--prompt-format',choices=['tokens','text','chat'],default='tokens')
    args=p.parse_args()
    cfg=json.loads((ROOT/'config.local.json').read_text())
    native=httpx.Client(base_url=cfg['upstream'],timeout=3600,trust_env=False,
        headers={'Authorization':'Bearer '+(ROOT/'state'/'internal-key.txt').read_text().strip()})
    seed=native.post('/tokenize',json={'content':' This is a stable benchmark sentence about rivers, trees, mathematics and science.','add_special':False}).json()['tokens']
    tokens=(seed*((args.prompt_tokens//len(seed))+1))[:args.prompt_tokens]
    if args.needle:
        messages=[
            {'role':'system','content':'Answer the final question using the document. Reply only with the passcode.'},
            {'role':'user','content':'The archive passcode is RIVER-7492.\nDocument:\n__BENCHMARK_FILLER__\nWhat is the archive passcode from the beginning?'}]
        template=native.post('/apply-template',json={'messages':messages,
            'chat_template_kwargs':{'enable_thinking':False}}).json()['prompt']
        before,after=template.split('__BENCHMARK_FILLER__')
        prefix=native.post('/tokenize',json={'content':before,'add_special':False,'parse_special':True}).json()['tokens']
        suffix=native.post('/tokenize',json={'content':after,'add_special':False,'parse_special':True}).json()['tokens']
        filler=tokens[:args.prompt_tokens-len(prefix)-len(suffix)]
        tokens=prefix+filler+suffix
    request_fields={'prompt':tokens}
    if args.prompt_format=='text':
        prompt_text=native.post('/detokenize',json={'tokens':tokens}).json()['content']
        assert native.post('/tokenize',json={'content':prompt_text,'add_special':False,'parse_special':True}).json()['tokens']==tokens
        request_fields={'prompt':prompt_text}
    elif args.prompt_format=='chat':
        assert args.needle,'Chat format requires --needle'
        content=native.post('/detokenize',json={'tokens':filler}).json()['content']
        messages[-1]['content']=messages[-1]['content'].replace('__BENCHMARK_FILLER__',content)
        rendered=native.post('/apply-template',json={'messages':messages,'chat_template_kwargs':{'enable_thinking':False}}).json()['prompt']
        assert native.post('/tokenize',json={'content':rendered,'add_special':False,'parse_special':True}).json()['tokens']==tokens
        request_fields={'messages':messages}
        # Synthetic payload for a separate replay test without benchmark-only tokenizer calls.
        fixture=ROOT/'build'/f'benchmark-chat-{args.prompt_tokens}.json'
        fixture.write_text(json.dumps(dict(synthetic_only=True,prompt_tokens=args.prompt_tokens,
            request=request_fields),ensure_ascii=False),encoding='utf-8')
    key=(ROOT/'state'/'api-key.txt').read_text().strip()
    client=OpenAI(base_url=f'http://{cfg["host"]}:{cfg["port"]}/v1',api_key=key,timeout=7200,max_retries=0,http_client=httpx.Client(trust_env=False))
    started_unix=time.time()
    start=time.perf_counter(); ttft=None; final={}; text=''
    create=client.chat.completions.create if args.prompt_format=='chat' else client.completions.create
    with create(model=cfg['model'],**request_fields,max_tokens=args.generate,temperature=0,seed=42,
            stream=True,stream_options={'include_usage':True},extra_body={'reasoning_effort':'none','ignore_eos':True,'session_id':args.session}) as stream:
        for chunk in stream:
            content=(chunk.choices[0].delta.content if args.prompt_format=='chat' else chunk.choices[0].text) if chunk.choices else ''
            if ttft is None and content:
                ttft=time.perf_counter()-start
            d=chunk.model_dump()
            if d.get('usage'): final['usage']=d['usage']
            if d.get('timings'): final['timings']=d['timings']
            if content: text+=content
    result=dict(started_unix=started_unix,finished_unix=time.time(),batch=cfg['batch'],ubatch=cfg['ubatch'],
        model=cfg['model'],
        spec_type=cfg.get('spec_type'),spec_draft_n_max=cfg.get('spec_draft_n_max') if cfg.get('spec_type') else None,
        spec_cache_k=cfg.get('spec_cache_type_k'),spec_cache_v=cfg.get('spec_cache_type_v'),
        runtime_manifest=cfg.get('runtime_manifest','runtime-built.json'),
        runtime_env=cfg.get('runtime_env',{}),profile_transfers=cfg.get('profile_transfers',False),
        session=args.session,prompt_format=args.prompt_format,forced_ignore_eos=True,
        context=cfg['context'],requested_prompt_tokens=args.prompt_tokens,requested_generation_tokens=args.generate,
        ttft_seconds=ttft,end_to_end_seconds=time.perf_counter()-start,devices=cfg['devices'],split=cfg['tensor_split'],
        cache_k=cfg['cache_type_k'],cache_v=cfg['cache_type_v'],**final)
    result['exact_prompt_verified']=final.get('usage',{}).get('prompt_tokens')==args.prompt_tokens
    if args.session:
        with httpx.Client(trust_env=False,timeout=30) as meta_client:
            meta=meta_client.get(f'http://{cfg["host"]}:{cfg["port"]}/sessions/{args.session}',
                headers={'Authorization':'Bearer '+key})
            meta.raise_for_status()
            result['session_cache']=meta.json()['last_request']
    if args.needle:
        result['needle_passed']='RIVER-7492' in text
        result['output']=text
    out=ROOT/'reports'/f'benchmark-{args.prompt_tokens}-{int(time.time())}.json'
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not result['exact_prompt_verified']:
        raise RuntimeError('Prompt token count mismatch or missing final usage')
    if args.needle and not result['needle_passed']:
        raise RuntimeError('Long-context needle retrieval failed')

if __name__=='__main__': main()
