"""In-Context Vectors (Liu et al. 2024).

Like FV but uses the residual stream (block output) instead of o_proj, and
scales by a single lambda. We sweep lambda and report the best.
"""
import time
import torch

from ..model_io import get_layers, free
from ..sensitivity import _forward_once


def collect_residual_delta_vectors(model, tokenizer, task, examples, demos,
                                   num_layers, max_seq_len, verbose=True):
    """Mean (block_out_fs - block_out_ns) at the last token of each example."""
    layers = get_layers(model)
    captured = [None] * num_layers

    def mk(li):
        def hook(_m, _i, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[li] = hs[:, -1, :].detach().float().cpu().clone()
        return hook

    handles = [layers[li].register_forward_hook(mk(li)) for li in range(num_layers)]
    deltas = {}
    n = len(examples)
    try:
        model.eval()
        for i, ex in enumerate(examples):
            _forward_once(model, tokenizer, task.build_prompt(ex, [], False), max_seq_len)
            z_ns = [captured[l] for l in range(num_layers)]
            free()
            _forward_once(model, tokenizer, task.build_prompt(ex, demos[i], True), max_seq_len)
            z_fs = [captured[l] for l in range(num_layers)]
            free()
            for l in range(num_layers):
                if z_ns[l] is not None and z_fs[l] is not None:
                    deltas.setdefault(l, []).append(z_fs[l] - z_ns[l])
            if verbose and (i + 1) % max(1, n // 4) == 0:
                print(f"  icv {i+1}/{n}", end="\r")
    finally:
        for h in handles:
            h.remove()

    return {l: torch.stack(xs, 0).mean(0).squeeze(0) for l, xs in deltas.items()}


class ICVInjector:
    """Add lam * delta_l to the last-token residual at every layer."""

    def __init__(self, model, deltas, lam):
        self.model = model
        self.deltas = deltas
        self.lam = float(lam)
        self.handles = []

    def __enter__(self):
        layers = get_layers(self.model)
        for l, d in self.deltas.items():
            delta_cpu = d.float().cpu()

            def mk(layer_idx, delta_cpu):
                def hook(_m, _i, output):
                    if isinstance(output, tuple):
                        y, rest = output[0], output[1:]
                    else:
                        y, rest = output, None
                    delta = delta_cpu.to(device=y.device, dtype=y.dtype)
                    if y.dim() == 3:
                        y = y.clone()
                        y[:, -1, :] = y[:, -1, :] + self.lam * delta
                    elif y.dim() == 2:
                        y = y + self.lam * delta.view(1, -1)
                    return ((y,) + rest) if rest is not None else y
                return hook

            h = layers[l].register_forward_hook(mk(l, delta_cpu))
            self.handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


DEFAULT_LAMBDA_GRID = (0.05, 0.10, 0.15, 0.20)


def run_icv_sweep(model, tokenizer, task, G_full, G_demos, eval_set,
                  num_layers, max_seq_len,
                  lambda_grid=DEFAULT_LAMBDA_GRID, verbose=True):
    from ..eval_engine import evaluate
    if verbose:
        print(f"  icv: collecting deltas on |G|={len(G_full)}")
    t0 = time.time()
    deltas = collect_residual_delta_vectors(model, tokenizer, task, G_full, G_demos,
                                            num_layers, max_seq_len, verbose=verbose)
    t_collect = time.time() - t0

    per_lam = {}
    best_acc, best_lam = -1.0, None
    for lam in lambda_grid:
        with ICVInjector(model, deltas, lam=lam):
            acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                                 [[] for _ in eval_set], max_seq_len, verbose=False)
        per_lam[lam] = acc
        if verbose:
            print(f"  icv lam={lam:.2f}  acc={acc*100:.2f}")
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    return {"best_acc": best_acc, "best_lambda": best_lam,
            "per_lambda": per_lam, "lambda_grid": list(lambda_grid),
            "n_calib": len(G_full), "collect_time_s": t_collect}
