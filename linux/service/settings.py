"""Expand installation-relative placeholders without publishing local paths."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def load_json(path, root=ROOT):
    def expand(value):
        if isinstance(value, str):
            return os.path.expandvars(value.replace('${ROOT}', str(root)))
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value
    return expand(json.loads(Path(path).read_text()))
