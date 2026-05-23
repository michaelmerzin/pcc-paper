"""Reproduce Table 1 of the paper.

Cross-architecture swept PCC results across 3 models × 8 tasks.

Usage:
    python experiments/table1_main_results.py \
        --model mistral7b --tasks sst2 mrpc dbpedia_14 boolq arc_challenge gsm8k mmlu ag_news \
        --out-dir runs/table1

    # For one task at a time on a single GPU (e.g. Kaggle T4):
    python experiments/table1_main_results.py --model mistral7b --tasks sst2

This script writes:
    runs/table1/<model>/<task>/result.json
    runs/table1/<model>/<task>/seed0_p_sweep.csv
    runs/table1/<model>/<task>/seed1_p_sweep.csv
    runs/table1/<model>/<task>/seed2_p_sweep.csv

Then run `scripts/aggregate_table1.py` to combine into the paper LaTeX table.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# Make `pcc` importable when this script is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.pipeline import run_pcc_multi_seed
from pcc.tasks import TASKS


def _result_to_json(r):
    """Convert PCCResult to JSON-serializable dict."""
    return {
        "model_key": r.model_key,
        "task_key": r.task_key,
        "fs_seed": r.fs_seed,
        "no_shot_acc": r.no_shot_acc,
        "few_shot_acc": r.few_shot_acc,
        "g_full_size": r.g_full_size,
        "n_calib": r.n_calib,
        "best_top_p": r.best_top_p,
        "best_top_acc": r.best_top_acc,
        "delta_fs": r.delta_fs,
        "top_neuron": list(r.top_neuron),
        "top_neuron_score": r.top_neuron_score,
        "top_accs": {f"{p:.2f}": v for p, v in r.top_accs.items()},
        "random_accs": {f"{p:.2f}": v for p, v in r.random_accs.items()},
        "t_total_s": r.t_total_s,
        "t_solve_avg_s": r.t_solve_avg_s,
    }


def run_one(model_key: str, task_key: str, out_root: str,
            n_eval: int, n_calib: int, fs_seeds, include_random: bool):
    """Run one (model, task) cell of Table 1 across all fs_seeds."""

    if task_key not in TASKS:
        raise KeyError(f"Unknown task: {task_key}")

    cfg = ExperimentConfig(
        model_key=model_key,
        task_key=task_key,
        n_eval=n_eval,
        n_calib=n_calib,
        fs_seeds=tuple(fs_seeds),
        out_dir=out_root,
        run_name=f"{model_key}/{task_key}",
    )
    cfg.finalize()

    print(f"\n{'=' * 70}")
    print(f"  TABLE 1 cell: {model_key} × {task_key}")
    print(f"{'=' * 70}\n")

    model, tokenizer = load_model_and_tokenizer(cfg)
    ORIG = snapshot_o_proj(model)

    try:
        results = run_pcc_multi_seed(model, tokenizer, cfg,
                                      weights_snapshot=ORIG,
                                      include_random_baseline=include_random,
                                      verbose=True)
    finally:
        restore_o_proj(model, ORIG)

    # ── Save per-seed JSON ───────────────────────────────────────────────────
    for r in results:
        path = os.path.join(cfg.out_dir, f"seed{r.fs_seed}.json")
        with open(path, "w") as f:
            json.dump(_result_to_json(r), f, indent=2)

        # CSV of the p-sweep
        rows = []
        for p in sorted(r.top_accs):
            row = {"p": p, "top_acc": r.top_accs[p]}
            if r.random_accs:
                row["random_mean"] = r.random_accs[p]["mean"]
                row["random_std"] = r.random_accs[p]["std"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(cfg.out_dir, f"seed{r.fs_seed}_p_sweep.csv"), index=False)

    # ── Aggregate across seeds (Table 1 row format) ──────────────────────────
    summary = {
        "model": model_key,
        "task": task_key,
        "n_seeds": len(results),
        "no_shot_mean": float(np.mean([r.no_shot_acc for r in results])),
        "no_shot_std": float(np.std([r.no_shot_acc for r in results])),
        "few_shot_mean": float(np.mean([r.few_shot_acc for r in results])),
        "few_shot_std": float(np.std([r.few_shot_acc for r in results])),
        "best_pcc_acc_mean": float(np.mean([r.best_top_acc for r in results])),
        "best_pcc_acc_std": float(np.std([r.best_top_acc for r in results])),
        "delta_fs_mean": float(np.mean([r.delta_fs for r in results])),
        "best_p_seed0": results[0].best_top_p,
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[summary] {model_key}/{task_key}:")
    print(f"  NS  = {summary['no_shot_mean']*100:.2f} ± {summary['no_shot_std']*100:.2f}%")
    print(f"  FS  = {summary['few_shot_mean']*100:.2f} ± {summary['few_shot_std']*100:.2f}%")
    print(f"  PCC = {summary['best_pcc_acc_mean']*100:.2f} ± {summary['best_pcc_acc_std']*100:.2f}%  "
          f"(best_p={summary['best_p_seed0']})")
    print(f"  ΔFS = {summary['delta_fs_mean']*100:+.2f}pp")

    # Free
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return summary


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 1 of the PCC paper.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Model key: mistral7b / phi3_mini / gemma2_2b")
    parser.add_argument("--tasks", nargs="+", required=True,
                        choices=list(TASKS.keys()),
                        help="One or more task keys to run.")
    parser.add_argument("--out-dir", default="runs/table1")
    parser.add_argument("--n-eval", type=int, default=150,
                        help="Eval-set size. Paper uses 150-500 depending on task.")
    parser.add_argument("--n-calib", type=int, default=64,
                        help="Calibration set size (paper default: 64).")
    parser.add_argument("--fs-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Few-shot demo seeds (paper default: 0 1 2).")
    parser.add_argument("--no-random-baseline", action="store_true",
                        help="Skip the random-row baseline (saves ~3x time per p).")
    args = parser.parse_args()

    all_summaries = []
    for tk in args.tasks:
        try:
            s = run_one(args.model, tk, args.out_dir,
                         n_eval=args.n_eval, n_calib=args.n_calib,
                         fs_seeds=args.fs_seeds,
                         include_random=not args.no_random_baseline)
            all_summaries.append(s)
        except Exception as e:
            print(f"\n[ERROR] {args.model}/{tk} failed: {e}")
            import traceback; traceback.print_exc()

    # Pretty-print the Table 1 fragment
    print("\n\n" + "=" * 80)
    print(f"  TABLE 1 — {args.model.upper()} (paper format)")
    print("=" * 80)
    print(f"{'Task':<16} {'NS':>8} {'FS':>10} {'Best PCC':>12} {'ΔFS':>8} {'Best p':>8}")
    print("-" * 80)
    for s in all_summaries:
        ns = s['no_shot_mean'] * 100
        fs = f"{s['few_shot_mean']*100:.2f}"
        if s['few_shot_std'] > 0.01:
            fs += f"±{s['few_shot_std']*100:.2f}"
        pcc = f"{s['best_pcc_acc_mean']*100:.2f}"
        if s['best_pcc_acc_std'] > 0.01:
            pcc += f"±{s['best_pcc_acc_std']*100:.2f}"
        print(f"{s['task']:<16} {ns:>7.2f} {fs:>10} {pcc:>12} "
              f"{s['delta_fs_mean']*100:>+7.2f} {s['best_p_seed0']:>8.2f}")


if __name__ == "__main__":
    main()