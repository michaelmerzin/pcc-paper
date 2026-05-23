"""Cross-fit no-sweep auto-sparsity selector (paper §3.4).

The idea:
    Sweeping p ∈ {0.05, ..., 0.30} requires running |topk_percents| separate
    evaluations to find the best p. The no-sweep variant uses no held-out
    evaluation at all — it picks p* purely from the activation deltas.

Algorithm:
    1. Split CALIB into two folds: A and B.
    2. Compute SCORES_A, H_NS_A, Z_FS_A on fold A. Same on fold B.
    3. Solve W*_S on fold A. Apply to fold B and measure the per-row
       coefficient of determination R²_l,j on the activation delta on B.
       R²_l,j = 1 - SS_res / SS_tot   (paper §3.4 line 297-302)
    4. Rank rows by R² and keep the prefix that covers 90% of the
       cumulative positive score → automatic sparsity p*.
    5. Refit ridge on the FULL CALIB at p*.

This adds roughly the cost of two extra ridge solves and ZERO evaluation
passes, making it ~10-20x faster than the oracle sweep (which requires
|topk_percents| full evaluation passes).
"""
from __future__ import annotations
from typing import Dict, Tuple, List
import torch

from .edit import solve_rows_closed_form


def compute_cross_fit_r2(
    H_NS: Dict[int, torch.Tensor],
    Z_FS: Dict[int, torch.Tensor],
    W_0: Dict[int, torch.Tensor],
    num_layers: int,
    hidden_size: int,
    mu: float,
) -> torch.Tensor:
    """Per-row held-out R² of the activation-delta prediction.

    For each layer l and each row j:
        1. Split (H_NS[l], Z_FS[l]) into folds A, B
        2. Solve W*_l on fold A
        3. Predict Z_fs on fold B as H_NS_B @ W*_l^T
        4. Compute R²_l,j = 1 - SS_res_B / SS_tot_B

    Returns:
        (L, hidden) tensor of R² values. Rows with high R² have a stable,
        learnable shift; rows with low/negative R² are noisy and should be skipped.
    """
    R2 = torch.zeros((num_layers, hidden_size), dtype=torch.float32)

    for l in range(num_layers):
        if l not in H_NS:
            continue
        H = H_NS[l].float()              # (N, d)
        Z = Z_FS[l].float()              # (N, d)
        W0 = W_0[l].float()              # (d_out=hidden, d_in=hidden)
        N = H.shape[0]
        if N < 4:                         # need at least 2 per fold
            continue

        half = N // 2
        H_A, H_B = H[:half], H[half:]
        Z_A, Z_B = Z[:half], Z[half:]

        # Solve on A (all rows at once)
        d = H_A.shape[1]
        A_mat = H_A.T @ H_A + mu * torch.eye(d)
        B_mat = Z_A.T @ H_A + mu * W0
        W_star = torch.linalg.solve(A_mat, B_mat.T).T   # (d_out, d)

        # Predict on B
        Z_B_pred = H_B @ W_star.T                          # (N_B, d_out)
        residual = Z_B - Z_B_pred                          # (N_B, d_out)
        ss_res = (residual ** 2).sum(dim=0)                # (d_out,)
        ss_tot = ((Z_B - Z_B.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
        ss_tot = torch.clamp(ss_tot, min=1e-8)
        R2[l] = 1.0 - ss_res / ss_tot

    return R2


def auto_select_rows_xdelta(
    H_NS: Dict[int, torch.Tensor],
    Z_FS: Dict[int, torch.Tensor],
    W_0: Dict[int, torch.Tensor],
    num_layers: int,
    hidden_size: int,
    mu: float,
    cumulative_target: float = 0.90,
) -> Tuple[List[Tuple[int, int]], float, torch.Tensor]:
    """No-sweep row selection via cross-fit explained-delta (paper §3.4).

    Returns:
        selected_rows : list of (layer, channel) to edit
        p_star        : automatic sparsity (fraction of total L*d rows)
        R2            : (L, hidden) R² scores (for diagnostics)
    """
    R2 = compute_cross_fit_r2(H_NS, Z_FS, W_0, num_layers, hidden_size, mu)

    # Positive R² only — negative means the prediction is worse than the mean
    pos_R2 = torch.clamp(R2, min=0.0)
    flat = pos_R2.view(-1)
    sorted_vals, sorted_idx = torch.sort(flat, descending=True)

    total = sorted_vals.sum().clamp(min=1e-8)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    # Number of rows needed to cover target fraction of explained delta
    n_take = int((cumsum / total <= cumulative_target).sum().item()) + 1
    n_take = max(1, min(n_take, len(flat)))

    selected_flat = sorted_idx[:n_take].tolist()
    selected_rows = [(i // hidden_size, i % hidden_size) for i in selected_flat]
    p_star = n_take / (num_layers * hidden_size)

    return selected_rows, p_star, R2
