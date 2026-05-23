"""High-level PCC pipeline — one call reproduces one Table 1 cell.

Flow (paper §3.1-3.3):
    1. Load splits: demo_pool, calib_pool, eval_set (strict, non-overlapping)
    2. Run no-shot eval on eval_set                          → NS accuracy
    3. Run few-shot eval on eval_set                         → FS accuracy (the teacher)
    4. Build G_full = {i : few-shot fixes no-shot}           (paper Eq. 1)
    5. Sample CALIB ⊆ G_full of size cfg.n_calib
    6. Compute SCORES = mean |z_fs - z_ns| over CALIB        (paper Eq. 2)
    7. Collect H_ns, Z_fs over CALIB                          (paper §3.3 inputs)
    8. For each sparsity p in cfg.topk_percents:
         a. Restore W to original
         b. select_rows(SCORES, p, mode='top')                (paper §3.2)
         c. apply_edit(...)                                    (paper Eq. 3-4)
         d. Re-evaluate on eval_set                            → PCC accuracy
    9. Best PCC = argmax over p.
"""
from __future__ import annotations
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

from .config import ExperimentConfig
from .tasks import get_task, effective_k_shots
from .eval_engine import evaluate, build_shots_per_example, build_correction_set
from .sensitivity import compute_sensitivity, collect_calibration_tensors
from .edit import select_rows, apply_edit
from .model_io import snapshot_o_proj, restore_o_proj


@dataclass
class PCCResult:
    """Result of one PCC run (one model × one task × one fs_seed × full p-sweep)."""
    model_key: str
    task_key: str
    fs_seed: int
    no_shot_acc: float
    few_shot_acc: float
    g_full_size: int
    n_calib: int

    # Per-sparsity results
    top_accs: Dict[float, float]            # p → accuracy
    random_accs: Dict[float, Dict]          # p → {'mean': ..., 'std': ...}

    # Best (argmax over p) — the headline number in Table 1
    best_top_p: float
    best_top_acc: float
    delta_fs: float

    # Sensitivity / localization metadata
    top_neuron: Tuple[int, int]             # (layer, channel) with highest score
    top_neuron_score: float

    # Timing
    t_total_s: float
    t_solve_avg_s: float


def run_pcc_one_seed(
    model, tokenizer, cfg: ExperimentConfig,
    fs_seed: int,
    weights_snapshot: Dict[int, torch.Tensor],
    include_random_baseline: bool = True,
    verbose: bool = True,
) -> PCCResult:
    """Run the full PCC pipeline once for one fs_seed.

    Caller is responsible for snapshotting the model BEFORE the call and
    restoring it AFTER (this function restores between p-sweep iterations).
    """
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    if cfg.max_seq_len is None:
        cfg.max_seq_len = task.recommended_max_seq_len

    use_canonical = (task.canonical_demos is not None
                     and k == len(task.canonical_demos)
                     and cfg.use_canonical_demos)

    t_start = time.time()

    # ── 1) Load splits ───────────────────────────────────────────────────────
    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    if verbose:
        kind = "canonical" if use_canonical else f"balanced seed={fs_seed}"
        print(f"\n[seed={fs_seed}] task={task.name}  k_shots={k}  ({kind})")
        print(f"  splits: demo_pool={len(demo_pool)}  calib_pool={len(calib_pool)}  eval={len(eval_set)}")

    # Build per-example shot lists
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, fs_seed, use_canonical)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, fs_seed, use_canonical)

    # ── 2) No-shot eval ──────────────────────────────────────────────────────
    if verbose: print(f"[seed={fs_seed}] no-shot eval...")
    ns_acc, ns_preds, _ = evaluate(model, tokenizer, task, eval_set,
                                    [[] for _ in eval_set], cfg.max_seq_len,
                                    bs=cfg.bs_eval, verbose=False)

    # ── 3) Few-shot eval ─────────────────────────────────────────────────────
    if verbose: print(f"[seed={fs_seed}] few-shot eval...")
    fs_acc, fs_preds, _ = evaluate(model, tokenizer, task, eval_set,
                                    eval_shots, cfg.max_seq_len,
                                    bs=cfg.bs_eval, verbose=False)
    if verbose:
        print(f"  eval-set: NS={ns_acc*100:.2f}%  FS={fs_acc*100:.2f}%")

    # ── 4) Build G_full from calib_pool ──────────────────────────────────────
    if verbose: print(f"[seed={fs_seed}] building G_full from calib_pool...")
    cp_ns_acc, cp_ns_preds, _ = evaluate(model, tokenizer, task, calib_pool,
                                          [[] for _ in calib_pool], cfg.max_seq_len,
                                          bs=cfg.bs_eval, verbose=False)
    cp_fs_acc, cp_fs_preds, _ = evaluate(model, tokenizer, task, calib_pool,
                                          calib_shots, cfg.max_seq_len,
                                          bs=cfg.bs_eval, verbose=False)
    g_idx = build_correction_set(calib_pool, cp_ns_preds, cp_fs_preds, task)
    G_full = [calib_pool[i] for i in g_idx]
    if verbose:
        print(f"  calib-pool: NS={cp_ns_acc*100:.2f}%  FS={cp_fs_acc*100:.2f}%  |G_full|={len(G_full)}")

    if len(G_full) == 0:
        raise RuntimeError(
            f"G_full is empty (no examples where few-shot corrects no-shot). "
            f"Increase n_calib_pool, or this task has a degenerate teacher.")

    # ── 5) CALIB ⊆ G_full ────────────────────────────────────────────────────
    n_cal = min(cfg.n_calib, len(G_full))
    rng_cal = random.Random(fs_seed * 17 + 3)
    cal_indices = list(range(len(G_full)))
    rng_cal.shuffle(cal_indices)
    cal_indices = cal_indices[:n_cal]
    CALIB = [G_full[i] for i in cal_indices]
    cal_demos = [calib_shots[g_idx[i]] for i in cal_indices]

    # ── 6) SCORES (paper Eq. 2) ──────────────────────────────────────────────
    if verbose: print(f"[seed={fs_seed}] computing SCORES on {len(CALIB)} examples...")
    SCORES = compute_sensitivity(model, tokenizer, task, CALIB, cal_demos,
                                  cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                  verbose=verbose)
    top_flat = SCORES.view(-1).argmax().item()
    top_neuron = (top_flat // cfg.hidden_size, top_flat % cfg.hidden_size)
    top_neuron_score = float(SCORES[top_neuron[0], top_neuron[1]])

    # ── 7) H_ns, Z_fs (inputs/outputs to o_proj at CALIB) ────────────────────
    if verbose: print(f"[seed={fs_seed}] collecting H_ns / Z_fs...")
    H_NS, Z_FS = collect_calibration_tensors(model, tokenizer, task, CALIB, cal_demos,
                                              cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                              verbose=verbose)

    # ── 8) Sweep over sparsity p (paper Table 1 search) ─────────────────────
    top_accs: Dict[float, float] = {}
    random_accs: Dict[float, Dict] = {}
    solve_times = []

    for p in cfg.topk_percents:
        # — TOP rows —
        restore_o_proj(model, weights_snapshot)
        t0 = time.time()
        sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size, seed=fs_seed)
        n_edited, _ = apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
        t_solve = time.time() - t0
        solve_times.append(t_solve)

        acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                              [[] for _ in eval_set], cfg.max_seq_len,
                              bs=cfg.bs_eval, verbose=False)
        top_accs[p] = acc
        if verbose:
            print(f"  [p={p:.2f}  top    ] acc={acc*100:.2f}%  ({n_edited} rows, solve={t_solve:.1f}s)")

        # — RANDOM rows (paper random-row baseline) —
        if include_random_baseline:
            r_accs = []
            for rs in cfg.rand_baseline_seeds:
                restore_o_proj(model, weights_snapshot)
                rsel = select_rows(SCORES, p, "random", cfg.num_layers, cfg.hidden_size, seed=rs)
                apply_edit(model, rsel, H_NS, Z_FS, cfg.mu, cfg.alpha)
                ra, _, _ = evaluate(model, tokenizer, task, eval_set,
                                     [[] for _ in eval_set], cfg.max_seq_len,
                                     bs=cfg.bs_eval, verbose=False)
                r_accs.append(ra)
            random_accs[p] = {"mean": float(np.mean(r_accs)), "std": float(np.std(r_accs))}
            if verbose:
                m, s = random_accs[p]["mean"], random_accs[p]["std"]
                print(f"  [p={p:.2f}  random ] acc={m*100:.2f}±{s*100:.2f}%")

    # Restore for the caller
    restore_o_proj(model, weights_snapshot)

    # ── 9) Best result ───────────────────────────────────────────────────────
    best_p = max(top_accs, key=top_accs.get)
    best_acc = top_accs[best_p]

    t_total = time.time() - t_start

    return PCCResult(
        model_key=cfg.model_key,
        task_key=cfg.task_key,
        fs_seed=fs_seed,
        no_shot_acc=ns_acc,
        few_shot_acc=fs_acc,
        g_full_size=len(G_full),
        n_calib=len(CALIB),
        top_accs=top_accs,
        random_accs=random_accs,
        best_top_p=best_p,
        best_top_acc=best_acc,
        delta_fs=best_acc - fs_acc,
        top_neuron=top_neuron,
        top_neuron_score=top_neuron_score,
        t_total_s=t_total,
        t_solve_avg_s=float(np.mean(solve_times)) if solve_times else 0.0,
    )


def run_pcc_multi_seed(
    model, tokenizer, cfg: ExperimentConfig,
    weights_snapshot: Optional[Dict[int, torch.Tensor]] = None,
    include_random_baseline: bool = True,
    verbose: bool = True,
) -> List[PCCResult]:
    """Run PCC across all fs_seeds in cfg.fs_seeds. Returns one PCCResult per seed."""
    if weights_snapshot is None:
        weights_snapshot = snapshot_o_proj(model)

    results = []
    for seed in cfg.fs_seeds:
        r = run_pcc_one_seed(model, tokenizer, cfg, seed, weights_snapshot,
                              include_random_baseline=include_random_baseline,
                              verbose=verbose)
        results.append(r)
        restore_o_proj(model, weights_snapshot)
    return results
