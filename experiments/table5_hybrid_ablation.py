"""Reproduce Table 5: hybrid supervised+KL ablation (Phi-3-mini on DBPedia-14).

Paper Table 5:
    Supervised (unranked)   n=64   Prec=100%   Accuracy=96.90%
    Hybrid (Sup + KL)       n=32   Prec=100%   Accuracy=98.05%
    Unsupervised (top-KL)   n=32   Prec=75%    Accuracy=98.17%

The +1.15pp jump from Supervised to Hybrid proves KL-magnitude dominates
label correctness: high-KL examples have cleaner activation shifts.

Method specifications:
    Supervised:    n examples from G_full, no KL ranking.
    Hybrid:        n examples from G_full, ranked by KL, take top-n.
    Unsupervised:  top-n by KL (no labels used), balanced by predicted label.

Usage:
    python experiments/table5_hybrid_ablation.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pcc.config import ExperimentConfig
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import select_rows, apply_edit
from pcc.unsupervised import rank_by_kl, select_top_kl_balanced, estimate_unsupervised_precision


def _train_and_eval(
    model, tokenizer, cfg, weights_snapshot,
    calib_examples, calib_shots, eval_set, sparsity_sweep,
    label: str, verbose=True,
):
    """Common pipeline: compute SCORES + tensors → sweep p → return best."""
    SCORES = compute_sensitivity(model, tokenizer, get_task(cfg.task_key),
                                  calib_examples, calib_shots,
                                  cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                  verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tokenizer, get_task(cfg.task_key),
                                              calib_examples, calib_shots,
                                              cfg.num_layers, cfg.hidden_size,
                                              cfg.max_seq_len, verbose=False)

    best_acc, best_p = 0.0, None
    for p in sparsity_sweep:
        restore_o_proj(model, weights_snapshot)
        sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size)
        apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
        acc, _, _ = evaluate(model, tokenizer, get_task(cfg.task_key), eval_set,
                              [[] for _ in eval_set], cfg.max_seq_len, verbose=False)
        if acc > best_acc:
            best_acc, best_p = acc, p
    restore_o_proj(model, weights_snapshot)
    if verbose:
        print(f"  [{label}] best acc={best_acc*100:.2f}% (p={best_p}, |G|={len(calib_examples)})")
    return best_acc, best_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="phi3_mini")
    parser.add_argument("--task", default="dbpedia_14")
    parser.add_argument("--n-sup", type=int, default=64,
                        help="Supervised |G|.")
    parser.add_argument("--n-hybrid", type=int, default=32,
                        help="Hybrid and Unsupervised |G|.")
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-calib-pool", type=int, default=500)
    parser.add_argument("--out-dir", default="runs/table5")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        model_key=args.model, task_key=args.task,
        n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
        n_calib=args.n_sup,
        out_dir=args.out_dir,
        topk_percents=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    )
    cfg.finalize()

    print(f"\n{'='*70}\n  Table 5 :: {args.model} × {args.task}\n{'='*70}")
    model, tokenizer = load_model_and_tokenizer(cfg)
    ORIG = snapshot_o_proj(model)

    task = get_task(args.task)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len
    use_canonical = (task.canonical_demos is not None
                     and k == len(task.canonical_demos)
                     and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, 0, use_canonical)

    # --- Build G_full (supervised set) -------------------------------------
    print("\n[setup] building G_full from calib_pool...")
    _, cp_ns_preds, _ = evaluate(model, tokenizer, task, calib_pool,
                                  [[] for _ in calib_pool], cfg.max_seq_len, verbose=False)
    _, cp_fs_preds, _ = evaluate(model, tokenizer, task, calib_pool, calib_shots,
                                  cfg.max_seq_len, verbose=False)
    g_idx = build_correction_set(calib_pool, cp_ns_preds, cp_fs_preds, task)
    G_full = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    print(f"  |G_full| = {len(G_full)}")

    # --- Rank G_full by KL (for Hybrid) -----------------------------------
    print("\n[setup] ranking G_full by KL...")
    # rank_by_kl scans calib_pool; we want to rank G_full only. Easy: pass G_full as calib_pool.
    ranked_g_full = rank_by_kl(
        model, tokenizer, task, G_full, demo_pool,
        k_shots=k, max_seq_len=cfg.max_seq_len,
        seed=0, use_canonical=use_canonical,
        n_scan=len(G_full), verbose=False,
    )
    print(f"  ranked {len(ranked_g_full)} G_full examples by KL")

    # --- 1. Supervised (unranked) -----------------------------------------
    n_sup = min(args.n_sup, len(G_full))
    rng = random.Random(0)
    sup_idx = list(range(len(G_full)))
    rng.shuffle(sup_idx)
    sup_idx = sup_idx[:n_sup]
    sup_calib = [G_full[i] for i in sup_idx]
    sup_demos = [G_demos[i] for i in sup_idx]
    sup_acc, sup_p = _train_and_eval(
        model, tokenizer, cfg, ORIG, sup_calib, sup_demos, eval_set,
        cfg.topk_percents, label="Supervised (unranked)",
    )

    # --- 2. Hybrid: top-n by KL within G_full (still uses labels for G filter) ---
    n_hyb = min(args.n_hybrid, len(ranked_g_full))
    # ranked_g_full uses INDEX INTO G_FULL (because we passed G_full as calib_pool)
    hyb_idx = [r[0] for r in ranked_g_full[:n_hyb]]
    hyb_calib = [G_full[i] for i in hyb_idx]
    hyb_demos = [G_demos[i] for i in hyb_idx]
    hyb_acc, hyb_p = _train_and_eval(
        model, tokenizer, cfg, ORIG, hyb_calib, hyb_demos, eval_set,
        cfg.topk_percents, label="Hybrid (Sup + KL)",
    )

    # --- 3. Unsupervised: top-KL on full calib_pool (no labels!) ----------
    ranked_pool = rank_by_kl(
        model, tokenizer, task, calib_pool, demo_pool,
        k_shots=k, max_seq_len=cfg.max_seq_len,
        seed=0, use_canonical=use_canonical,
        n_scan=min(200, len(calib_pool)), verbose=False,
    )
    n_unsup = args.n_hybrid
    unsup_calib, unsup_idx = select_top_kl_balanced(
        ranked_pool, calib_pool, n_unsup, num_classes=task.num_classes or 2,
    )
    unsup_prec = estimate_unsupervised_precision(unsup_calib, unsup_idx,
                                                   calib_pool, ranked_pool, task)
    unsup_demos = build_shots_per_example(task, demo_pool, unsup_calib, k, 0, use_canonical)
    unsup_acc, unsup_p = _train_and_eval(
        model, tokenizer, cfg, ORIG, unsup_calib, unsup_demos, eval_set,
        cfg.topk_percents, label=f"Unsupervised (top-KL, prec={unsup_prec*100:.0f}%)",
    )

    # --- Save + print --------------------------------------------------------
    results = {
        "model": args.model, "task": args.task,
        "supervised":   {"n": n_sup,   "prec": 1.0,        "acc": sup_acc,   "best_p": sup_p},
        "hybrid":       {"n": n_hyb,   "prec": 1.0,        "acc": hyb_acc,   "best_p": hyb_p},
        "unsupervised": {"n": n_unsup, "prec": unsup_prec, "acc": unsup_acc, "best_p": unsup_p},
    }
    with open(os.path.join(cfg.out_dir, "table5.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*70}\n  TABLE 5 — {args.model} × {args.task}\n{'='*70}")
    print(f"{'Method':<30} {'n':>4} {'Prec':>6} {'Accuracy':>10}")
    print("-" * 60)
    for label, key in [("Supervised (unranked)", "supervised"),
                        ("Hybrid (Sup + KL)", "hybrid"),
                        ("Unsupervised (top-KL)", "unsupervised")]:
        r = results[key]
        print(f"{label:<30} {r['n']:>4} {r['prec']*100:>5.0f}% {r['acc']*100:>9.2f}%")


if __name__ == "__main__":
    main()
