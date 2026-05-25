"""Ridge-anchored o_proj row edit:

    W_S* = argmin || H W^T - Z_S ||_F^2 + mu || W - W0_S ||_F^2

then blend back:  W_new = (1 - alpha) W0 + alpha W*.  Sparsity p controls how
many rows are edited (k = p * L * d).
"""
import random
import torch

from .model_io import get_layers


def solve_rows_closed_form(H_ns, Z_fs_S, W0_S, mu):
    # H: (N, d), Y: (N, k), W0: (k, d)
    H = H_ns.float()
    Y = Z_fs_S.float()
    W0 = W0_S.float()
    d = H.shape[1]
    A = H.T @ H + mu * torch.eye(d, dtype=torch.float32)
    B = Y.T @ H + mu * W0
    return torch.linalg.solve(A, B.T).T  # (k, d)


def select_rows(scores, p, mode, num_layers, hidden_size, seed=0, layer_subset=None):
    """Return list of (layer, channel) to edit. k = round(p * L * d)."""
    L, H = num_layers, hidden_size
    k = max(1, int(round(p * L * H)))

    if mode == "top":
        idx = torch.topk(scores.view(-1), k).indices.cpu().tolist()
        return [(i // H, i % H) for i in idx]

    if mode == "random":
        rng = random.Random(seed)
        rows = [(l, n) for l in range(L) for n in range(H)]
        rng.shuffle(rows)
        return rows[:k]

    if mode in ("early", "late"):
        if layer_subset is None:
            layer_subset = (tuple(range(L // 2)) if mode == "early"
                            else tuple(range(L // 2, L)))
        mask = torch.full_like(scores, float("-inf"))
        for l in layer_subset:
            mask[l] = scores[l]
        k = min(k, len(layer_subset) * H)
        idx = torch.topk(mask.view(-1), k).indices.cpu().tolist()
        return [(i // H, i % H) for i in idx]

    raise ValueError(f"bad mode: {mode!r}")


def apply_edit(model, selected_rows, H_ns, Z_fs, mu, alpha):
    """Edit in-place. Caller is responsible for snapshot/restore."""
    by_layer = {}
    for l, n in selected_rows:
        by_layer.setdefault(l, []).append(n)

    per_layer = {}
    n_edited = 0
    layers = get_layers(model)

    for l, rows in by_layer.items():
        rows = sorted(set(rows))
        per_layer[l] = len(rows)

        W = layers[l].self_attn.o_proj.weight
        dev, dt = W.device, W.dtype

        H = H_ns[l]                                # (N, d)
        Y = Z_fs[l][:, rows]                       # (N, k)
        W0 = W.data[rows].detach().cpu().float()   # (k, d)

        W_star = solve_rows_closed_form(H, Y, W0, mu)
        W_new = (1.0 - alpha) * W0 + alpha * W_star

        W.data[rows] = W_new.to(dev, dtype=dt)
        n_edited += len(rows)

    return n_edited, per_layer
