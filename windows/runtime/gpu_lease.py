"""Cross-process exclusion between the LLM and speech GPU supervisors."""
import msvcrt
from pathlib import Path


class GpuLease:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(__file__).resolve().parent / 'gpu-service.lock'
        self.handle = None

    def acquire(self):
        if self.handle is not None:
            raise RuntimeError('GPU lease already held by this object')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open('a+b')
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            handle.close()
            raise RuntimeError('Another GPU service is running. Stop it before starting LLM or speech.') from exc
        self.handle = handle
        return self

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.close()
