"""Observe SSE timing without buffering delivery or retaining prompt/output text."""
import json
import time

class StreamTiming:
    def __init__(self, began):
        self.began = began
        self.pending = b''
        self.first_token_ms = None
        self.first_text_ms = None

    def observe(self, chunk):
        if self.first_token_ms is not None and self.first_text_ms is not None:
            return
        self.pending = (self.pending + chunk).replace(b'\r\n', b'\n')
        while b'\n\n' in self.pending:
            event, self.pending = self.pending.split(b'\n\n', 1)
            raw = b'\n'.join(line[5:].lstrip() for line in event.split(b'\n') if line.startswith(b'data:'))
            if not raw or raw == b'[DONE]':
                continue
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            text = reasoning = tools = False
            for choice in value.get('choices', []):
                delta = choice.get('delta') or {}
                text |= bool(choice.get('text') or delta.get('content'))
                reasoning |= bool(delta.get('reasoning') or delta.get('reasoning_content'))
                functions = [x.get('function', {}) for x in delta.get('tool_calls', [])]
                functions.append(delta.get('function_call') or {})
                tools |= any(bool(x.get('name') or x.get('arguments')) for x in functions)
            kind = value.get('type', '')
            text |= kind == 'response.output_text.delta' and bool(value.get('delta'))
            reasoning |= kind in ('response.reasoning_text.delta', 'response.reasoning_summary_text.delta') and bool(value.get('delta'))
            tools |= kind == 'response.function_call_arguments.delta' and bool(value.get('delta'))
            elapsed = (time.monotonic() - self.began) * 1000
            if (text or reasoning or tools) and self.first_token_ms is None:
                self.first_token_ms = elapsed
            if text and self.first_text_ms is None:
                self.first_text_ms = elapsed
        # Bound observer memory for non-SSE or malformed upstream content.
        if len(self.pending) > 1048576:
            self.pending = b''
