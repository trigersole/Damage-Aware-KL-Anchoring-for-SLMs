"""Immutable identities and committed recovery for the isolated calibration."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen identity changed; preserve output and use a new experiment ID: ' + str(path))
    else:
        atomic(path, value)


@contextmanager
def lease(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def rng_state():
    import torch
    import numpy as np
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    import numpy as np
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def parameters(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(parameters(model).items()):
        h.update(name.encode()); h.update(str((value.shape, value.dtype)).encode())
        h.update(value.contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def tree_hash(value):
    """Stable tensor-aware digest for recovery equality checks, not identity config."""
    import torch
    import numpy as np
    h = hashlib.sha256()
    def walk(x):
        if isinstance(x, torch.Tensor):
            data = x.detach().cpu().contiguous()
            h.update(str((data.dtype, tuple(data.shape))).encode())
            h.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda k: (type(k).__name__, str(k))):
                walk(k); walk(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x:
                walk(item)
        else:
            h.update((type(x).__name__ + ':' + repr(x) + ';').encode())
    walk(value)
    return h.hexdigest()


def save(folder, model, optimizer, scheduler, progress, identity):
    import torch
    generation = 'recovery-' + uuid.uuid4().hex
    path = folder / 'recovery' / generation; path.mkdir(parents=True)
    torch.save({'identity': identity, 'parameters': parameters(model),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'progress': progress, 'rng': rng_state()}, path / 'state.pt')
    # Fsync the completed payload before atomically committing its pointer.
    with (path / 'state.pt').open('rb') as stream:
        os.fsync(stream.fileno())
    atomic(folder / 'recovery/current.json', {'generation': generation,
                                            'sha256': sha(path / 'state.pt')})
    # Only our disposable recovery generations are pruned. Scientific adapters,
    # previous experiment folders and committed result receipts are not removed.
    previous = sorted((folder / 'recovery').glob('recovery-*'), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in previous[2:]:
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old)
    atomic(folder / 'training_history.json', progress['history'])
    atomic(folder / 'validation_history.json', progress['validation_history'])


def load(folder, model, optimizer, scheduler, identity):
    import torch
    pointer = read(folder / 'recovery/current.json')
    generation = pointer['generation']
    if not generation.startswith('recovery-') or Path(generation).name != generation:
        raise ValueError('Invalid recovery directory')
    path = folder / 'recovery' / generation / 'state.pt'
    if sha(path) != pointer['sha256']:
        raise ValueError('Recovery checksum differs')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['identity'] != identity:
        raise ValueError('Recovery identity/hardware/software differs; preserve state and inspect')
    current = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(current) != set(payload['parameters']):
        raise ValueError('Recovery trainable parameter keys differ')
    with torch.no_grad():
        for name, value in payload['parameters'].items():
            if current[name].shape != value.shape:
                raise ValueError('Recovery parameter shape differs')
            current[name].copy_(value)
    optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return payload['progress']


def checkpoint(folder, model, tokenizer, step, identity):
    destination = folder / 'checkpoints' / f'step_{step:06d}'
    if destination.exists():
        receipt = read(destination / 'receipt.json')
        if receipt['identity'] != identity or receipt['step'] != step:
            raise ValueError('Existing checkpoint identity differs')
        if any(sha(destination / f) != h for f, h in receipt['files'].items()):
            raise ValueError('Existing checkpoint changed')
        return
    tmp = folder / ('checkpoint-staging-' + uuid.uuid4().hex); tmp.mkdir()
    model.save_pretrained(tmp, safe_serialization=True); tokenizer.save_pretrained(tmp)
    files = {p.name: sha(p) for p in tmp.iterdir() if p.is_file()}
    atomic(tmp / 'receipt.json', {'identity': identity, 'step': step, 'files': files})
    destination.parent.mkdir(exist_ok=True)
    os.rename(tmp, destination)


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic(); self.seconds = seconds; self.requested = False

    def signal(self, *_):
        self.requested = True

    def expired(self):
        return self.requested or time.monotonic() - self.started >= self.seconds
