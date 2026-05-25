"""U[-1,1] random fill on the same selected rows PCC edits. Expected to collapse."""
import torch

from ..model_io import get_layers


def apply_random_fill(model, selected_rows, seed=0, low=-1.0, high=1.0):
    gen = torch.Generator().manual_seed(seed)
    by_layer = {}
    for l, n in selected_rows:
        by_layer.setdefault(l, []).append(n)

    n_edited = 0
    for l, rows in by_layer.items():
        rows = sorted(set(rows))
        W = get_layers(model)[l].self_attn.o_proj.weight
        dev, dt = W.device, W.dtype
        noise = torch.empty((len(rows), W.shape[1]))
        noise.uniform_(low, high, generator=gen)
        W.data[rows] = noise.to(dev, dtype=dt)
        n_edited += len(rows)
    return n_edited
