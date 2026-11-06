"""Map request reasoning controls to the pinned Swift chat template."""

EFFORTS = {'none': 'none', 'minimal': 'low', 'low': 'low',
           'medium': 'medium', 'high': 'xhigh', 'xhigh': 'xhigh'}


def prepare(body, responses=False):
    options = body.get('chat_template_kwargs')
    nested = body.get('reasoning')
    effort = nested.get('effort') if responses and isinstance(nested, dict) else body.get('reasoning_effort')
    if effort is None and isinstance(options, dict):
        effort = options.get('reasoning_effort')
    if effort is None and not (isinstance(options, dict) and options.get('enable_thinking') is True):
        return body
    if options is not None and not isinstance(options, dict):
        raise ValueError('chat_template_kwargs must be an object')
    options = dict(options or {})
    if effort is None:
        effort = 'xhigh'
    if not isinstance(effort, str) or effort not in EFFORTS:
        raise ValueError('Supported reasoning effort: none, minimal, low, medium, high, xhigh')
    canonical = EFFORTS[effort]
    options['reasoning_effort'] = canonical
    if canonical == 'none':
        options['enable_thinking'] = False
    else:
        options.setdefault('enable_thinking', True)
    result = dict(body, chat_template_kwargs=options)
    if responses and isinstance(nested, dict) and nested.get('effort') is not None:
        result['reasoning'] = dict(nested, effort=canonical)
    elif 'reasoning_effort' in body:
        result['reasoning_effort'] = canonical
    return result
