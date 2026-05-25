"""Cross-fit auto-p selector. The idea is to pick the sparsity without any
held-out evaluation:

  - split CALIB into two halves A and B
  - solve W* on A
  - compute per-row R^2 on B (how well the predicted activation delta matches)
  - take rows in decreasing R^2 until cumulative explained delta hits 90%

This adds two ridge solves and zero eval passes — much cheaper than a real sweep.
"""
import torch


def compute_cross_fit_r2(H_NS, Z_FS, W0, num_layers, hidden_size, mu):
    R2 = torch.zeros((num_layers, hidden_size), dtype=torch.float32)

    for l in range(num_layers):
        if l not in H_NS:
            continue
        H = H_NS[l].float()
        Z = Z_FS[l].float()
        W0l = W0[l].float()
        N = H.shape[0]
        if N < 4:
            continue   # need at least 2 per fold

        half = N // 2
        H_A, H_B = H[:half], H[half:]
        Z_A, Z_B = Z[:half], Z[half:]

        d = H_A.shape[1]
        A_mat = H_A.T @ H_A + mu * torch.eye(d)
        B_mat = Z_A.T @ H_A + mu * W0l
        W_star = torch.linalg.solve(A_mat, B_mat.T).T

        Z_B_pred = H_B @ W_star.T
        resid = Z_B - Z_B_pred
        ss_res = (resid ** 2).sum(dim=0)
        ss_tot = ((Z_B - Z_B.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp(min=1e-8)
        R2[l] = 1.0 - ss_res / ss_tot
    return R2


def auto_select_rows_xdelta(H_NS, Z_FS, W0, num_layers, hidden_size, mu,
                            cumulative_target=0.90):
    R2 = compute_cross_fit_r2(H_NS, Z_FS, W0, num_layers, hidden_size, mu)
    pos = torch.clamp(R2, min=0.0)  # negative R^2 means worse than the mean
    flat = pos.view(-1)
    sorted_vals, sorted_idx = torch.sort(flat, descending=True)

    total = sorted_vals.sum().clamp(min=1e-8)
    cum = torch.cumsum(sorted_vals, dim=0)
    n_take = int((cum / total <= cumulative_target).sum().item()) + 1
    n_take = max(1, min(n_take, len(flat)))

    sel = sorted_idx[:n_take].tolist()
    rows = [(i // hidden_size, i % hidden_size) for i in sel]
    p_star = n_take / (num_layers * hidden_size)
    return rows, p_star, R2
