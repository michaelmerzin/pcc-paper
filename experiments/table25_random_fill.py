"""Reproduce Table 25: Random-weight ablation (paper §6.2).

Replace the closed-form solution W*_S with U[-1, 1] noise into the IDENTICAL
selected rows. Paper claim:

    Variant                                Acc       Δ NS      Δ FS
    No-shot baseline (seed 1)             78.44%      —      −6.77pp
    Few-shot teacher (seed 1)             85.21%   +6.77pp     —
    PCC closed-form at p=0.25             89.07%  +10.63pp   +3.86pp
    Random U[−1, 1] fill (all p)           0.00%  −78.44pp  −85.21pp

The 0.00% is expected: weight magnitudes are O(1e-2), U[-1,1] is O(1) → 100x
too large → saturates the residual stream → degenerate generation → 0% accuracy.

Usage:
    python experiments/table25_random_fill.py --model mistral7b --task dbpedia_14
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import select_rows, apply_edit
from pcc.baselines.random_fill import apply_random_fill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral7b")
    parser.add_argument("--task", default="dbpedia_14")
    parser.add_argument("--sparsities", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--fill-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="RNG seeds for U[-1,1] (paper uses 3 fill seeds).")
    parser.add_argument("--fs-seed", type=int, default=1,
                        help="Few-shot demo seed (paper Table 25 uses seed 1).")
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-calib-pool", type=int, default=500)
    parser.add_argument("--out-dir", default="runs/table25")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        model_key=args.model, task_key=args.task,
        n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
        out_dir=args.out_dir,
    )
    cfg.finalize()

    print(f"\n{'='*70}\n  Table 25 :: {args.model} × {args.task}\n{'='*70}")
    model, tokenizer = load_model_and_tokenizer(cfg)
    ORIG = snapshot_o_proj(model)

    task = get_task(args.task)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len
    use_canonical = (task.canonical_demos is not None
                     and k == len(task.canonical_demos)
                     and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, args.fs_seed, use_canonical)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, args.fs_seed, use_canonical)

    # ── Baselines ────────────────────────────────────────────────────────────
    print("\n[baselines]")
    ns_acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                              [[] for _ in eval_set], cfg.max_seq_len, verbose=False)
    fs_acc, _, _ = evaluate(model, tokenizer, task, eval_set, eval_shots,
                              cfg.max_seq_len, verbose=False)
    print(f"  No-shot baseline (seed {args.fs_seed}) : {ns_acc*100:.2f}%")
    print(f"  Few-shot teacher (seed {args.fs_seed}) : {fs_acc*100:.2f}%")

    # ── Build G_full + CALIB for PCC ─────────────────────────────────────────
    _, cp_ns, _ = evaluate(model, tokenizer, task, calib_pool,
                            [[] for _ in calib_pool], cfg.max_seq_len, verbose=False)
    _, cp_fs, _ = evaluate(model, tokenizer, task, calib_pool, calib_shots,
                            cfg.max_seq_len, verbose=False)
    g_idx = build_correction_set(calib_pool, cp_ns, cp_fs, task)
    G_full = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    print(f"  |G_full| = {len(G_full)}")

    # SCORES & calib tensors on G_full (PCC will select rows from here)
    SCORES = compute_sensitivity(model, tokenizer, task, G_full, G_demos,
                                  cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tokenizer, task, G_full, G_demos,
                                              cfg.num_layers, cfg.hidden_size,
                                              cfg.max_seq_len, verbose=False)

    # ── 1. PCC closed-form at the paper-reported best p (0.25 for Mistral DBPedia) ─
    print("\n[1. PCC closed-form]")
    pcc_results = {}
    for p in args.sparsities:
        restore_o_proj(model, ORIG)
        sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size)
        apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
        acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                              [[] for _ in eval_set], cfg.max_seq_len, verbose=False)
        pcc_results[p] = acc
        print(f"  PCC at p={p:.2f} : {acc*100:.2f}%")
    best_pcc_p = max(pcc_results, key=pcc_results.get)
    best_pcc_acc = pcc_results[best_pcc_p]
    print(f"  → best p={best_pcc_p:.2f}, acc={best_pcc_acc*100:.2f}%")

    # ── 2. Random U[-1, 1] fill — same rows, replaced with noise ─────────
    print("\n[2. Random U[-1,1] fill — same rows as PCC]")
    rand_results = {}
    for p in args.sparsities:
        accs = []
        for fs in args.fill_seeds:
            restore_o_proj(model, ORIG)
            sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size)
            apply_random_fill(model, sel, seed=fs, low=-1.0, high=1.0)
            acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                                  [[] for _ in eval_set], cfg.max_seq_len, verbose=False)
            accs.append(acc)
        m, s = float(np.mean(accs)), float(np.std(accs))
        rand_results[p] = {"mean": m, "std": s, "n_seeds": len(accs)}
        print(f"  Random fill at p={p:.2f} : {m*100:.2f}±{s*100:.2f}%")

    restore_o_proj(model, ORIG)

    # ── Save & report (paper Table 25 format) ───────────────────────────────
    results = {
        "model": args.model, "task": args.task, "fs_seed": args.fs_seed,
        "no_shot_acc": ns_acc, "few_shot_acc": fs_acc,
        "pcc_at_each_p": {f"{p:.2f}": v for p, v in pcc_results.items()},
        "pcc_best_p": best_pcc_p, "pcc_best_acc": best_pcc_acc,
        "random_fill_at_each_p": {f"{p:.2f}": v for p, v in rand_results.items()},
    }
    with open(os.path.join(cfg.out_dir, "table25.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Pretty-print Table 25
    print(f"\n\n{'='*70}\n  TABLE 25 — {args.model} × {args.task}\n{'='*70}")
    print(f"{'Variant':<40} {'Acc':>10} {'Δ NS':>10} {'Δ FS':>10}")
    print("-" * 70)
    print(f"{'No-shot baseline (seed ' + str(args.fs_seed) + ')':<40} "
          f"{ns_acc*100:>9.2f}% {'—':>10} {(ns_acc-fs_acc)*100:>+9.2f}pp")
    print(f"{'Few-shot teacher (seed ' + str(args.fs_seed) + ')':<40} "
          f"{fs_acc*100:>9.2f}% {(fs_acc-ns_acc)*100:>+9.2f}pp {'—':>10}")
    print(f"{'PCC closed-form at p=' + f'{best_pcc_p:.2f}':<40} "
          f"{best_pcc_acc*100:>9.2f}% {(best_pcc_acc-ns_acc)*100:>+9.2f}pp "
          f"{(best_pcc_acc-fs_acc)*100:>+9.2f}pp")
    avg_rand = float(np.mean([v["mean"] for v in rand_results.values()]))
    print(f"{'Random U[-1,1] fill (avg over p)':<40} "
          f"{avg_rand*100:>9.2f}% {(avg_rand-ns_acc)*100:>+9.2f}pp "
          f"{(avg_rand-fs_acc)*100:>+9.2f}pp")


if __name__ == "__main__":
    main()
