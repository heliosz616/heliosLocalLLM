from prompt_cache import preparation_key

def test_sampling_and_transport_do_not_change_prompt_identity():
    a={'messages':[{'role':'user','content':'你好'}],'reasoning_effort':'medium','temperature':0.2,'max_tokens':128,'stream':True}
    b=dict(a,temperature=0.9,max_tokens=1024,stream=False,stream_options={'include_usage':True})
    assert preparation_key('/v1/chat/completions',a)==preparation_key('/v1/chat/completions',b)

def test_template_inputs_and_endpoint_must_invalidate_identity():
    a={'messages':[{'role':'user','content':'你好'}],'reasoning_effort':'medium'}
    key=preparation_key('/v1/chat/completions',a)
    for change in ({'reasoning_effort':'none'},{'messages':[{'role':'user','content':'再见'}]},
                   {'tools':[{'type':'function'}]},{'chat_template_kwargs':{'enable_thinking':False}},
                   {'tool_choice':'required'}):
        assert preparation_key('/v1/chat/completions',dict(a,**change))!=key
    assert preparation_key('/v1/completions',a)!=key
