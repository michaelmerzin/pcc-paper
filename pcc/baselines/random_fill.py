"""Random-weight fill ablation (paper §6.2, Table 25).

Replace the closed-form solution W*_S with U[-1, 1] noise into the IDENTICAL
selected rows. Paper claim: accuracy collapses to 0.00% across all sparsity
ranges (p ∈ {0.05 ... 0.30}) for all seeds.

Why 0.00% rather than 1/C (random-guessing baseline)?
    Natural weight magnitudes in attention output projections are O(1e-2).
    U[-1, 1] noise is roughly two orders of magnitude larger. Even at the
    smallest tested sparsity (p=0.05, ~6,500 rows on Mistral-7B), this
    saturates the residual stream and prevents coherent token generation,
    producing degenerate outputs that match no valid class label.

This is the cleanest possible isolation of "the closed-form solve matters"
vs "row selection matters": same rows, only the values change.
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import torch

from ..model_io import get_layers


def apply_random_fill(
    model,
    selected_rows: List[Tuple[int, int]],
    seed: int = 0,
    low: float = -1.0,
    high: float = 1.0,
):
    """Replace selected o_proj rows with U[low, high] noise in-place.

    Args:
        model:           HF causal LM (edited in place; caller must restore)
        selected_rows:   list of (layer, row) — same rows the supervised PCC edit uses
        seed:            RNG seed for reproducibility
        low, high:       Uniform distribution endpoints (paper uses -1, 1)
    """
    gen = torch.Generator().manual_seed(seed)

    # Group by layer
    by_layer: Dict[int, List[int]] = {}
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
