"""Standard-library storage, allocation budget and single-writer helpers."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    # Unicode paragraph separators can occur inside a valid JSON string.
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_file(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def jsonl(path, values):
    with Path(path).open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        json_file(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lease(output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another preflight writer is active") from None
        yield


class Pause(Exception):
    pass


class Budget:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or not 0 < seconds <= 19800:
            raise ValueError("Use a positive budget of at most 5.5 hours")
        self.start, self.seconds, self.requested = time.monotonic(), seconds, False
        self.handlers = {}

    def __enter__(self):
        for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.handlers[number] = signal.getsignal(number)
            signal.signal(number, self.stop)
        return self

    def stop(self, *_):
        self.requested = True

    def check(self):
        if self.requested or time.monotonic() - self.start >= self.seconds:
            raise Pause("Budget/signal pause; repeat the same sbatch command after this job ends")

    def __exit__(self, *_):
        for number, handler in self.handlers.items():
            signal.signal(number, handler)
