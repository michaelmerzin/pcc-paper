"""Reproduce appendix Tables 9-23: per-task per-model capacity sweeps.

Each table is a sweep of p ∈ {0.05, 0.10, ..., 0.30} for one (model, task)
showing Top accuracy vs Random accuracy. The paper appendix has:

    Mistral-7B   : Tables 9 (AG News), 10 (DBPedia), 11 (SST-2), 12 (MRPC),
                   13 (ARC-Challenge), 14 (GSM8K)
    Phi-3-mini   : Tables 15 (GSM8K), 16 (MRPC), 17 (ARC-Challenge),
                   18 (AG News)
    Gemma-2-2B   : Tables 19 (SST-2), 20 (MRPC), 21 (AG News),
                   22 (DBPedia), 23 (ARC-Challenge)

This script is identical to Table 1's pipeline but writes the full per-p
sweep (not just the best) into a CSV per (model, task). The Table 1 script
already does this — this script is a convenience wrapper that uses the
appropriate sparsity grid (paper uses 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
0.30 for Gemma-2-2B and 0.05-0.30 in steps of 0.05 for the others).

Usage:
    python experiments/tables_9_23_capacity_sweeps.py --model mistral7b --task ag_news
    python experiments/tables_9_23_capacity_sweeps.py --model gemma2_2b --task sst2  # uses 0.075 too
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.pipeline import run_pcc_multi_seed
from pcc.tasks import TASKS


# Paper-reported sparsity grids per model (Gemma uses a finer grid)
SPARSITY_GRIDS = {
    "mistral7b": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    "phi3_mini": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    "gemma2_2b": (0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--task",  required=True, choices=list(TASKS))
    parser.add_argument("--out-dir", default="runs/capacity_sweeps")
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--fs-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    sparsity_grid = SPARSITY_GRIDS[args.model]
    cfg = ExperimentConfig(
        model_key=args.model, task_key=args.task,
        n_eval=args.n_eval, n_calib=args.n_calib,
        fs_seeds=tuple(args.fs_seeds),
        topk_percents=sparsity_grid,
        out_dir=args.out_dir,
        run_name=f"{args.model}/{args.task}",
    )
    cfg.finalize()

    print(f"\n{'='*70}\n  Capacity sweep :: {args.model} × {args.task}\n"
          f"  sparsity grid: {sparsity_grid}\n{'='*70}")

    model, tokenizer = load_model_and_tokenizer(cfg)
    ORIG = snapshot_o_proj(model)
    try:
        results = run_pcc_multi_seed(model, tokenizer, cfg,
                                      weights_snapshot=ORIG,
                                      include_random_baseline=True,
                                      verbose=True)
    finally:
        restore_o_proj(model, ORIG)

    # ── Aggregate per-p across seeds ────────────────────────────────────────
    rows = []
    for p in sparsity_grid:
        top_accs = [r.top_accs[p] for r in results if p in r.top_accs]
        if not top_accs:
            continue
        row = {
            "p": p,
            "top_mean": sum(top_accs) / len(top_accs),
            "top_std": (sum((x - sum(top_accs)/len(top_accs))**2 for x in top_accs) / len(top_accs))**0.5,
        }
        if all(p in r.random_accs for r in results):
            r_means = [r.random_accs[p]["mean"] for r in results]
            r_stds = [r.random_accs[p]["std"] for r in results]
            row["random_mean"] = sum(r_means) / len(r_means)
            row["random_std"] = max(r_stds)
            row["gap"] = row["top_mean"] - row["random_mean"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.out_dir, "sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[saved] {csv_path}")

    # Paper-format pretty-print
    print(f"\n{'='*70}\n  CAPACITY SWEEP — {args.model} × {args.task}\n{'='*70}")
    if "random_mean" in df.columns:
        print(f"{'p':>6} {'Top acc':>14} {'Random acc':>14} {'Gap':>8}")
        for _, r in df.iterrows():
            print(f"{r['p']:>6.3f} {r['top_mean']*100:>9.2f}±{r['top_std']*100:>3.2f} "
                  f"{r['random_mean']*100:>9.2f}±{r['random_std']*100:>3.2f} "
                  f"{(r['top_mean']-r['random_mean'])*100:>+6.2f}")
    else:
        print(f"{'p':>6} {'Top acc':>10}")
        for _, r in df.iterrows():
            print(f"{r['p']:>6.3f} {r['top_mean']*100:>9.2f}")


if __name__ == "__main__":
    main()
