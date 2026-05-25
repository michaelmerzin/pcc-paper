"""Main flow: NS/FS eval, build G, sensitivity, ridge solve, sweep p, return best."""
import random
import time

import numpy as np
import torch

from .config import ExperimentConfig
from .tasks import get_task, effective_k_shots
from .eval_engine import evaluate, build_shots_per_example, build_correction_set
from .sensitivity import compute_sensitivity, collect_calibration_tensors
from .edit import select_rows, apply_edit
from .model_io import snapshot_o_proj, restore_o_proj


def run_pcc_one_seed(model, tokenizer, cfg, fs_seed, weights_snapshot,
                    include_random_baseline=True, verbose=True):
    """Returns a dict with NS/FS/top_accs/random_accs/best, etc. Restores
    weights after the p-sweep, so the caller can run another seed cleanly."""

    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    if cfg.max_seq_len is None:
        cfg.max_seq_len = task.recommended_max_seq_len

    use_canon = (task.canonical_demos is not None
                 and k == len(task.canonical_demos)
                 and cfg.use_canonical_demos)

    t_start = time.time()

    # splits
    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    if verbose:
        kind = "canon" if use_canon else f"balanced seed={fs_seed}"
        print(f"\nseed {fs_seed} :: {task.name}  k={k}  ({kind})")
        print(f"  pools: demo={len(demo_pool)} calib={len(calib_pool)} eval={len(eval_set)}")

    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, fs_seed, use_canon)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, fs_seed, use_canon)

    # NS / FS on eval
    if verbose: print(f"  ns eval...")
    ns_acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                            [[] for _ in eval_set], cfg.max_seq_len, bs=cfg.bs_eval)
    if verbose: print(f"  fs eval...")
    fs_acc, _, _ = evaluate(model, tokenizer, task, eval_set, eval_shots,
                            cfg.max_seq_len, bs=cfg.bs_eval)
    if verbose:
        print(f"  ns={ns_acc*100:.2f}  fs={fs_acc*100:.2f}")

    # build G on calib_pool
    if verbose: print(f"  building G...")
    _, cp_ns_preds, _ = evaluate(model, tokenizer, task, calib_pool,
                                 [[] for _ in calib_pool], cfg.max_seq_len, bs=cfg.bs_eval)
    _, cp_fs_preds, _ = evaluate(model, tokenizer, task, calib_pool, calib_shots,
                                 cfg.max_seq_len, bs=cfg.bs_eval)
    g_idx = build_correction_set(calib_pool, cp_ns_preds, cp_fs_preds, task)
    G_full = [calib_pool[i] for i in g_idx]
    if verbose:
        print(f"  |G_full| = {len(G_full)}")

    if len(G_full) == 0:
        raise RuntimeError(
            "G_full empty (FS doesn't fix NS anywhere). "
            "Bump n_calib_pool or check the task is non-degenerate.")

    # CALIB <= G_full
    n_cal = min(cfg.n_calib, len(G_full))
    rng_cal = random.Random(fs_seed * 17 + 3)
    cal_order = list(range(len(G_full)))
    rng_cal.shuffle(cal_order)
    cal_order = cal_order[:n_cal]
    CALIB = [G_full[i] for i in cal_order]
    cal_demos = [calib_shots[g_idx[i]] for i in cal_order]

    # sensitivity
    if verbose: print(f"  sensitivity on |CALIB|={len(CALIB)}")
    SCORES = compute_sensitivity(model, tokenizer, task, CALIB, cal_demos,
                                 cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                 verbose=verbose)
    top_flat = SCORES.view(-1).argmax().item()
    top_neuron = (top_flat // cfg.hidden_size, top_flat % cfg.hidden_size)
    top_score = float(SCORES[top_neuron[0], top_neuron[1]])

    # H_ns / Z_fs
    if verbose: print(f"  collecting H_ns / Z_fs")
    H_NS, Z_FS = collect_calibration_tensors(model, tokenizer, task, CALIB, cal_demos,
                                             cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                             verbose=verbose)

    # sweep p
    top_accs = {}
    random_accs = {}
    solve_times = []

    for p in cfg.topk_percents:
        restore_o_proj(model, weights_snapshot)
        t0 = time.time()
        sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size, seed=fs_seed)
        n_edit, _ = apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
        t_solve = time.time() - t0
        solve_times.append(t_solve)

        acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                             [[] for _ in eval_set], cfg.max_seq_len, bs=cfg.bs_eval)
        top_accs[p] = acc
        if verbose:
            print(f"  p={p:.2f} top:    acc={acc*100:.2f}  rows={n_edit}  solve={t_solve:.1f}s")

        if include_random_baseline:
            r_accs = []
            for rs in cfg.rand_baseline_seeds:
                restore_o_proj(model, weights_snapshot)
                rsel = select_rows(SCORES, p, "random", cfg.num_layers, cfg.hidden_size, seed=rs)
                apply_edit(model, rsel, H_NS, Z_FS, cfg.mu, cfg.alpha)
                ra, _, _ = evaluate(model, tokenizer, task, eval_set,
                                    [[] for _ in eval_set], cfg.max_seq_len, bs=cfg.bs_eval)
                r_accs.append(ra)
            random_accs[p] = {"mean": float(np.mean(r_accs)), "std": float(np.std(r_accs))}
            if verbose:
                m, s = random_accs[p]["mean"], random_accs[p]["std"]
                print(f"  p={p:.2f} random: acc={m*100:.2f}±{s*100:.2f}")

    restore_o_proj(model, weights_snapshot)

    best_p = max(top_accs, key=top_accs.get)
    best_acc = top_accs[best_p]

    return {
        "model_key": cfg.model_key,
        "task_key": cfg.task_key,
        "fs_seed": fs_seed,
        "no_shot_acc": ns_acc,
        "few_shot_acc": fs_acc,
        "g_full_size": len(G_full),
        "n_calib": len(CALIB),
        "top_accs": top_accs,
        "random_accs": random_accs,
        "best_top_p": best_p,
        "best_top_acc": best_acc,
        "delta_fs": best_acc - fs_acc,
        "top_neuron": top_neuron,
        "top_neuron_score": top_score,
        "t_total_s": time.time() - t_start,
        "t_solve_avg_s": float(np.mean(solve_times)) if solve_times else 0.0,
    }


def run_pcc_multi_seed(model, tokenizer, cfg, weights_snapshot=None,
                       include_random_baseline=True, verbose=True):
    if weights_snapshot is None:
        weights_snapshot = snapshot_o_proj(model)
    out = []
    for seed in cfg.fs_seeds:
        r = run_pcc_one_seed(model, tokenizer, cfg, seed, weights_snapshot,
                             include_random_baseline=include_random_baseline,
                             verbose=verbose)
        out.append(r)
        restore_o_proj(model, weights_snapshot)
    return out
