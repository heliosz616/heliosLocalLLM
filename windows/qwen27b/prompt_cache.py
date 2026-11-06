"""Character SHA for prompt preparation, independent of generation-only controls."""
import hashlib,json

GENERATION_FIELDS=frozenset(('max_tokens','max_completion_tokens','temperature','top_p','top_k',
    'min_p','typical_p','seed','repeat_penalty','repetition_penalty','presence_penalty',
    'frequency_penalty','ignore_eos','stream','stream_options'))

def preparation_key(path,body):
    prompt_fields={k:v for k,v in body.items() if k not in GENERATION_FIELDS}
    serialized=json.dumps(prompt_fields,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256((path+'\n'+serialized).encode('utf-8')).hexdigest()
