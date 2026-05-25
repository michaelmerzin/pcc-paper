"""Cross-fit auto-sparsity selector (X-Delta-CAP).

For each o_proj row, fit the ridge on one fold of the calibration pool
and measure held-out R^2 on the other. Score each row by
max(0, R^2) * Var(delta) and keep the shortest prefix covering
`cumulative_target` of the total score, subject to:
  - late-layer restriction (l >= L // 2)
  - sparsity clamp p in [P_MIN, P_MAX]
  - per-layer cap LAYER_CAP_FRAC * d rows

Matches Section 3.3 of the paper.
"""
import random
from typing import Dict, List, Tuple

import torch


# Hyperparameters from the paper.
P_MIN = 0.05
P_MAX = 0.3
LAYER_CAP_FRAC = 0.65
CUMULATIVE_TARGET = 0.90


def _ridge_solve_chunk(H_fit: torch.Tensor, Y_fit_chunk: torch.Tensor,
                       mu: float) -> torch.Tensor:
    """Solve (H^T H + mu I) X = H^T Y for a chunk of output columns."""
    A = H_fit.T @ H_fit
    A.diagonal().add_(float(mu))
    B = H_fit.T @ Y_fit_chunk

    try:
        chol = torch.linalg.cholesky(A)
    except Exception:
        A.diagonal().add_(1e-3)
        chol = torch.linalg.cholesky(A)

    X = torch.cholesky_solve(B, chol)
    return X.T  # shape: (chunk, d)


def auto_select_rows_xdelta(
    H_NS: Dict[int, torch.Tensor],
    Z_FS: Dict[int, torch.Tensor],
    W0: Dict[int, torch.Tensor],
    num_layers: int,
    hidden_size: int,
    mu: float,
    cumulative_target: float = CUMULATIVE_TARGET,
    chunk_rows: int = 512,
    seed: int = 0,
    device: str = "cuda",
    verbose: bool = False,
) -> Tuple[List[Tuple[int, int]], float, Dict[Tuple[int, int], float]]:
    """Cross-fit selector matching the paper.

    Returns:
        selected: list of (layer, row) pairs
        p_star:   selected_count / (L * d)
        R2_map:   {(layer, row): score} for diagnostics
    """
    L = int(num_layers)
    d = int(hidden_size)

    # ----- 1. derive observed deltas -----
    # Z_FS is the with-context o_proj output; H_NS @ W0^T is the no-context
    # output, so delta = Z_FS - (H_NS @ W0^T).
    Z_NS = {}
    for l in range(L):
        if l in H_NS and l in W0:
            Z_NS[l] = (H_NS[l].float() @ W0[l].float().T).cpu()

    # ----- 2. score rows on late layers only -----
    layers_to_use = list(range(L // 2, L))

    all_rows: List[Tuple[float, int, int]] = []
    per_layer_total_score: Dict[int, float] = {}

    for l in layers_to_use:
        if l not in H_NS or l not in Z_FS or l not in Z_NS:
            continue

        H = H_NS[l].float()
        Y = (Z_FS[l].float() - Z_NS[l].float())  # observed delta, shape (m, d)
        m, d_check = H.shape
        assert d_check == d, (d_check, d)

        # Split calibration examples into two folds.
        rng = random.Random(seed + 1009 * l)
        idx = list(range(m))
        rng.shuffle(idx)
        half = m // 2
        idx_a = torch.tensor(idx[:half], dtype=torch.long)
        idx_b = torch.tensor(idx[half:], dtype=torch.long)

        if len(idx_a) < 2 or len(idx_b) < 2:
            continue

        layer_score = torch.zeros(d, dtype=torch.float32)
        folds = [(idx_a, idx_b), (idx_b, idx_a)]

        for fit_idx, val_idx in folds:
            H_fit = H[fit_idx].to(device)
            Y_fit = Y[fit_idx].to(device)
            H_val = H[val_idx].to(device)
            Y_val = Y[val_idx].to(device)

            # SS_tot per column = |B| * Var(delta) (up to the 1/|B| factor;
            # we keep it in this form so score = max(0,R2) * Var(delta) drops
            # straight out of max(0, SS_tot - SS_res) / |B|).
            y_mean = Y_val.mean(dim=0, keepdim=True)
            sst = ((Y_val - y_mean) ** 2).sum(dim=0) + 1e-8

            fold_score = torch.zeros(d, dtype=torch.float32, device="cpu")

            for start in range(0, d, chunk_rows):
                end = min(start + chunk_rows, d)

                DeltaW_chunk = _ridge_solve_chunk(
                    H_fit=H_fit, Y_fit_chunk=Y_fit[:, start:end], mu=mu,
                )
                pred = H_val @ DeltaW_chunk.T

                residual = Y_val[:, start:end] - pred
                sse = (residual ** 2).sum(dim=0)
                sst_chunk = sst[start:end]

                r2 = 1.0 - sse / sst_chunk
                r2_pos = torch.clamp(r2, min=0.0)

                # max(0, R2) * Var(delta), modulo the 1/|B| factor on Var.
                # Constant factor doesn't affect ranking, but keep it
                # equivalent for transparency.
                var_delta = sst_chunk / max(1, len(val_idx))
                score = r2_pos * var_delta

                fold_score[start:end] = score.detach().float().cpu()

            layer_score += fold_score / len(folds)

        total_score = float(layer_score.sum().item())
        per_layer_total_score[l] = total_score

        for j in range(d):
            s = float(layer_score[j].item())
            if s > 0:
                all_rows.append((s, l, j))

        if verbose:
            print(f"[auto-p] layer={l:02d} "
                  f"total_score={total_score:.4e} "
                  f"max_row={float(layer_score.max().item()):.4e}")

    if not all_rows:
        raise RuntimeError("auto-p: no positive-scoring rows found")

    # ----- 3. shortest 90% prefix -----
    all_rows.sort(reverse=True, key=lambda x: x[0])

    scores = torch.tensor([x[0] for x in all_rows], dtype=torch.float32)
    cum = torch.cumsum(scores, dim=0) / (scores.sum() + 1e-12)
    k_energy = int(torch.searchsorted(cum, torch.tensor(cumulative_target)).item()) + 1

    # ----- 4. sparsity clamp -----
    total_rows = L * d
    min_k = int(round(P_MIN * total_rows))
    max_k = int(round(P_MAX * total_rows))
    k = max(k_energy, min_k)
    k = min(k, max_k, len(all_rows))

    # ----- 5. per-layer cap -----
    layer_cap = int(round(LAYER_CAP_FRAC * d))
    selected: List[Tuple[int, int]] = []
    layer_counts: Dict[int, int] = {}
    R2_map: Dict[Tuple[int, int], float] = {}

    for score, l, j in all_rows:
        if len(selected) >= k:
            break
        if layer_counts.get(l, 0) >= layer_cap:
            continue
        selected.append((l, j))
        layer_counts[l] = layer_counts.get(l, 0) + 1
        R2_map[(l, j)] = score

    p_star = len(selected) / float(total_rows)

    if verbose:
        print(f"[auto-p] k_energy={k_energy}  k_after_clamp={k}  "
              f"layer_cap={layer_cap}  p*={p_star:.4f}")
        print(f"[auto-p] selected per-layer: "
              f"{sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)}")

    return selected, p_star, R2_map