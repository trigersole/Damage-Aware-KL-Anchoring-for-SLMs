"""Deterministic FP64 cosine farthest-first with resumable numeric state."""
import hashlib
import math


def farthest_first(ids, embeddings, *, seed, device, check, publish, resume=None, every=128):
    import numpy as np
    import torch
    if type(seed) is not int or type(every) is not int or every < 1:
        raise ValueError('Invalid seed/checkpoint interval')
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(k, str) or not k for k in ids):
        raise ValueError('IDs must be unique nonempty strings')
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError('Embedding shape/values differ')
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    keys = [ids[i] for i in order]
    x = torch.as_tensor(np.asarray(matrix[order], dtype=np.float64), device=device)
    norms = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError('Nonfinite/zero vector norm')
    x = x / norms
    n = len(keys)
    first = min(range(n), key=lambda i: (hashlib.sha256(f'{seed}\0{keys[i]}'.encode()).digest(), keys[i]))
    prefix = [] if resume is None else resume['prefix_indices'][:]
    if (any(type(i) is not int or not 0 <= i < n for i in prefix)
            or len(set(prefix)) != len(prefix) or (prefix and prefix[0] != first)):
        raise ValueError('Invalid saved ranking prefix')
    if prefix:
        values = resume['nearest_distances']
        if len(values) != n or any(not math.isfinite(v) or not -1e-12 <= v <= 2 + 1e-12 for v in values):
            raise ValueError('Invalid saved nearest distances')
        nearest = torch.tensor(values, dtype=torch.float64, device=device)
    else:
        if resume is not None and resume['nearest_distances'] is not None:
            raise ValueError('Empty prefix must have no saved distances')
        nearest = torch.full((n,), float('inf'), dtype=torch.float64, device=device)
    selected = torch.zeros(n, dtype=torch.bool, device=device)
    if prefix:
        selected[prefix] = True

    def state():
        return {'prefix_indices': prefix[:], 'nearest_distances': nearest.cpu().tolist() if prefix else None}

    for rank in range(len(prefix), n):
        try:
            check()
        except Exception:
            publish(state())
            raise
        if rank == 0:
            chosen = first
        else:
            best = torch.max(nearest[~selected])
            tied = (~selected) & torch.isclose(nearest, best, rtol=0., atol=1e-12)
            chosen = int(torch.nonzero(tied, as_tuple=False)[0, 0].item())
        selected[chosen] = True
        prefix.append(chosen)
        distance = 1. - torch.clamp(x @ x[chosen], -1., 1.)
        nearest = torch.minimum(nearest, distance)
        if len(prefix) % every == 0 or len(prefix) == n:
            publish(state())
    return {keys[index]: rank for rank, index in enumerate(prefix)}
