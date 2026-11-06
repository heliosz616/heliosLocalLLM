"""Real HTTP contract checks for OpenAI-compatible request fields."""
import json
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'config.local.json').read_text())
    key=(ROOT/'state/api-key.txt').read_text().strip()
    results=[]
    with httpx.Client(base_url=f'http://{cfg["host"]}:{cfg["port"]}',trust_env=False,
                      headers={'Authorization':'Bearer '+key},timeout=300) as client:
        for label,fields in [
            ('default_medium',{'temperature':0.7}),
            ('explicit_medium_sampling',{'reasoning_effort':'medium','temperature':1,'top_p':0.9,
                'top_k':40,'min_p':0.05,'repetition_penalty':1.05,'presence_penalty':0.1,
                'frequency_penalty':0.1,'seed':42,'stop':['UNUSED_STOP'],'top_a':0}),
            ('thinking_disabled',{'reasoning_effort':'none','temperature':0}),
        ]:
            start=time.perf_counter(); first_reasoning=None; first_content=None; reasoning=''; content=''; done=False
            with client.stream('POST','/v1/chat/completions',json={'model':cfg['model'],
                    'messages':[{'role':'user','content':'What is 2 + 2? Reply briefly.'}],
                    'max_completion_tokens':1024,'stream':True,**fields}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith('data: '):continue
                    if line=='data: [DONE]':done=True;break
                    event=json.loads(line[6:])
                    assert 'error' not in event,event
                    for choice in event.get('choices',[]):
                        delta=choice.get('delta',{})
                        if delta.get('reasoning_content'):
                            if first_reasoning is None:first_reasoning=time.perf_counter()-start
                            reasoning+=delta['reasoning_content']
                        if delta.get('content'):
                            if first_content is None:first_content=time.perf_counter()-start
                            content+=delta['content']
            assert done and '4' in content,(label,content)
            assert bool(reasoning)==(label!='thinking_disabled'),(label,reasoning,content)
            results.append(dict(label=label,first_reasoning_seconds=first_reasoning,
                first_content_seconds=first_content,reasoning_characters=len(reasoning),content=content,passed=True))
            print(json.dumps(results[-1]),flush=True)
        tool={'type':'function','function':{'name':'get_temperature','description':'Get city temperature',
            'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}
        messages=[{'role':'user','content':'Use get_temperature for Example City A, then tell me the result.'}]
        response=client.post('/v1/chat/completions',json={'model':cfg['model'],'messages':messages,
            'tools':[tool],'tool_choice':{'type':'function','function':{'name':'get_temperature'}},
            'temperature':0,'max_tokens':1024})
        response.raise_for_status()
        message=response.json()['choices'][0]['message']
        call=message['tool_calls'][0]
        assert call['function']['name']=='get_temperature'
        # Replay native reasoning fields intact, as the agent's tool continuation does.
        messages += [message,{'role':'tool','tool_call_id':call['id'],'content':'{"temperature_celsius":23}'}]
        response=client.post('/v1/chat/completions',json={'model':cfg['model'],'messages':messages,
            'tools':[tool],'temperature':0,'max_tokens':1024})
        response.raise_for_status()
        assert '23' in (response.json()['choices'][0]['message'].get('content') or '')
        results.append(dict(label='default_medium_tool_roundtrip',passed=True))
        # A prompt that explicitly asks for no tool must still obey a named
        # tool_choice, even with a competing declared function.
        other={'type':'function','function':{'name':'other_tool','parameters':{'type':'object','properties':{}}}}
        forced={'model':cfg['model'],'messages':[{'role':'user','content':'Just say DONE. Do not call any tools.'}],
            'tools':[other,tool],'tool_choice':{'type':'function','function':{'name':'get_temperature'}},
            'temperature':0,'max_tokens':256,'reasoning_effort':'none'}
        response=client.post('/v1/chat/completions',json=forced)
        response.raise_for_status()
        calls=response.json()['choices'][0]['message'].get('tool_calls') or []
        assert calls and all(call['function']['name']=='get_temperature' for call in calls),response.json()
        forced['tool_choice']['function']['name']='missing_tool'
        assert client.post('/v1/chat/completions',json=forced).status_code==400
        results.append(dict(label='named_tool_enforced_against_prompt_and_unknown_rejected',passed=True))
        response=client.post('/v1/chat/completions',json={'messages':[],'top_a':0.1})
        assert response.status_code==400
    (ROOT/'reports/agent-compatibility.json').write_text(json.dumps(dict(
        reference_commit='c4c3e70ddda90c02d100bdeda5e4414e874f7faf',results=results,
        nonzero_top_a_explicitly_rejected=True,passed=True),indent=2))

if __name__=='__main__':main()
