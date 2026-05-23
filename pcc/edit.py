"""Closed-form ridge-anchored edit on attention o_proj rows.

This is the mathematical core of PCC (paper §3.3, Equations 3 and 4):

    W_S^* = argmin_W   ||H^ns W^T - Z^fs_S||_F^2  +  μ ||W - W_{0,S}||_F^2

    closed-form solution:
    W_S^* = [(H^ns^T H^ns + μ I)^{-1} (H^ns^T Z^fs_S + μ W_{0,S}^T)]^T

We then blend back to the original weights (paper §3.3, line 289-290):
    W_new = (1 - α) W_0 + α W^*

Sparsity p sets how many rows are edited:  k = p * L * d  (paper §3.2, line 280-281).
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import random
import torch

from .model_io import get_layers


# ============================================================================ #
#   Paper Eq. 4 — closed-form ridge-anchored solve on selected rows           #
# ============================================================================ #
def solve_rows_closed_form(
    H_ns: torch.Tensor,
    Z_fs_S: torch.Tensor,
    W_0_S: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """Compute W_S^* using the closed-form ridge-anchored solution.

    Args:
        H_ns:    (N, d)   no-shot inputs to o_proj at this layer
        Z_fs_S:  (N, k)   few-shot outputs of o_proj at the SELECTED rows
        W_0_S:   (k, d)   original weight rows being edited
        mu:               anchor strength (paper uses 1e-2)

    Returns:
        (k, d) updated rows.
    """
    H = H_ns.float()
    Y = Z_fs_S.float()
    W0 = W_0_S.float()
    d = H.shape[1]

    A = H.T @ H + mu * torch.eye(d, dtype=torch.float32)   # (d, d)
    B = Y.T @ H + mu * W0                                   # (k, d)
    # Symmetric A: solve A X^T = B^T, return X = (k, d)
    return torch.linalg.solve(A, B.T).T


# ============================================================================ #
#   Paper §3.2 — sensitivity-based row selection                              #
# ============================================================================ #
def select_rows(
    scores: torch.Tensor,            # (L, d) sensitivity SCORES
    p: float,                         # sparsity fraction
    mode: str,                        # 'top' | 'random' | 'early' | 'late'
    num_layers: int,
    hidden_size: int,
    seed: int = 0,
    layer_subset: Optional[Tuple[int, ...]] = None,
) -> List[Tuple[int, int]]:
    """Return list of (layer, channel) tuples to edit, of length k ≈ p*L*d.

    Modes:
        'top'    — top-p by sensitivity score (paper main method)
        'random' — random rows, seeded (paper random-row baseline)
        'early'  — top-p restricted to early layers (ablation)
        'late'   — top-p restricted to late layers (ablation)
    """
    L, H = num_layers, hidden_size
    k = max(1, int(round(p * L * H)))

    if mode == "top":
        flat = scores.view(-1)
        idx = torch.topk(flat, k).indices.cpu().tolist()
        return [(i // H, i % H) for i in idx]

    if mode == "random":
        rng = random.Random(seed)
        all_rows = [(l, n) for l in range(L) for n in range(H)]
        rng.shuffle(all_rows)
        return all_rows[:k]

    if mode in ("early", "late"):
        if layer_subset is None:
            layer_subset = (tuple(range(L // 2)) if mode == "early"
                            else tuple(range(L // 2, L)))
        mask = torch.full_like(scores, float("-inf"))
        for l in layer_subset:
            mask[l] = scores[l]
        flat = mask.view(-1)
        k_cap = min(k, len(layer_subset) * H)
        idx = torch.topk(flat, k_cap).indices.cpu().tolist()
        return [(i // H, i % H) for i in idx]

    raise ValueError(f"Unknown selection mode: {mode!r}")


# ============================================================================ #
#   Paper §3.3 line 289-290 — apply the blended edit to selected rows         #
# ============================================================================ #
def apply_edit(
    model,
    selected_rows: List[Tuple[int, int]],
    H_ns: Dict[int, torch.Tensor],
    Z_fs: Dict[int, torch.Tensor],
    mu: float,
    alpha: float,
) -> Tuple[int, Dict[int, int]]:
    """Apply  W_new = (1-α) W_0 + α W*  to the selected o_proj rows in-place.

    Args:
        model:          HF causal LM (edited in place)
        selected_rows:  list of (layer, row) from select_rows()
        H_ns[l]:        (N, d) no-shot inputs to o_proj at layer l
        Z_fs[l]:        (N, d) few-shot outputs of o_proj at layer l (FULL d, we slice)
        mu, alpha:      paper hyperparameters (1e-2, 0.8)

    Returns:
        (n_edited, per_layer_count)
    """
    # Group rows by layer
    by_layer: Dict[int, List[int]] = {}
    for l, n in selected_rows:
        by_layer.setdefault(l, []).append(n)

    per_layer_count = {}
    n_edited = 0

    for l, rows in by_layer.items():
        rows = sorted(set(rows))
        per_layer_count[l] = len(rows)

        block = get_layers(model)[l]
        W = block.self_attn.o_proj.weight       # (d_out, d_in) = (hidden, hidden)
        dev, dt = W.device, W.dtype

        H = H_ns[l]                              # (N, d)
        Y = Z_fs[l][:, rows]                     # (N, k) — slice to selected rows
        W0 = W.data[rows].detach().cpu().float() # (k, d)

        W_star = solve_rows_closed_form(H, Y, W0, mu)     # (k, d)
        W_new = (1.0 - alpha) * W0 + alpha * W_star        # (k, d)

        W.data[rows] = W_new.to(dev, dtype=dt)
        n_edited += len(rows)

    return n_edited, per_layer_count
