from pathlib import Path
import pytest
import json
import subprocess
import sys

def render(**values):
    code = "import json,sys; from jinja2 import Environment; d=json.load(sys.stdin); print(Environment().from_string(d.pop('template')).render(**d),end='')"
    return subprocess.check_output([sys.executable, '-c', code], input=json.dumps(dict(template=TEMPLATE.read_text(), **values)), text=True)

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates/openai-compatible-bare.jinja'

@pytest.mark.parametrize('effort', ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'])
def test_no_synthetic_system_instruction(effort):
    text = render(messages=[{'role': 'user', 'content': 'hello'}],
                           enable_thinking=effort != 'none', reasoning_effort=effort,
                           add_generation_prompt=True)
    assert '<|im_start|>system' not in text
    assert 'Reasoning effort is set' not in text
    assert '<|im_start|>user\nhello<|im_end|>' in text


def test_preserve_explicit_system_and_tools():
    messages = [{'role':'system','content':'User-owned instruction.'}, {'role':'user','content':'hello'}]
    text = render(messages=messages, enable_thinking=True, reasoning_effort='xhigh',
                           tools=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object'}}}],
                           add_generation_prompt=True)
    assert text.count('User-owned instruction.') == 1
    assert 'lookup' in text and '<tool_call>' in text
    assert 'Reasoning effort is set' not in text
