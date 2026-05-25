"""Per-task capacity sweep: write the full p-sweep CSV instead of just the best.

Same pipeline as main_results but with a different sparsity grid for Gemma
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.pipeline import run_pcc_multi_seed
from pcc.tasks import TASKS


SPARSITY_GRIDS = {
    "mistral7b": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    "phi3_mini": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    "gemma2_2b": (0.050, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--out-dir", default="runs/capacity")
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--fs-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    grid = SPARSITY_GRIDS[args.model]
    cfg = ExperimentConfig(
        model_key=args.model, task_key=args.task,
        n_eval=args.n_eval, n_calib=args.n_calib,
        fs_seeds=tuple(args.fs_seeds),
        topk_percents=grid,
        out_dir=args.out_dir,
        run_name=f"{args.model}/{args.task}")
    cfg.finalize()

    print(f"\n{'='*70}\n  capacity :: {args.model} x {args.task}\n  grid: {grid}\n{'='*70}")
    model, tok = load_model_and_tokenizer(cfg)
    ORIG = snapshot_o_proj(model)
    try:
        results = run_pcc_multi_seed(model, tok, cfg,
                                     weights_snapshot=ORIG,
                                     include_random_baseline=True)
    finally:
        restore_o_proj(model, ORIG)

    # per-p aggregation across seeds
    rows = []
    for p in grid:
        tops = [r["top_accs"][p] for r in results if p in r["top_accs"]]
        if not tops:
            continue
        mean_top = sum(tops) / len(tops)
        std_top = (sum((x - mean_top) ** 2 for x in tops) / len(tops)) ** 0.5
        row = {"p": p, "top_mean": mean_top, "top_std": std_top}
        if all(p in r["random_accs"] for r in results):
            r_means = [r["random_accs"][p]["mean"] for r in results]
            r_stds = [r["random_accs"][p]["std"] for r in results]
            row["random_mean"] = sum(r_means) / len(r_means)
            row["random_std"] = max(r_stds)
            row["gap"] = row["top_mean"] - row["random_mean"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.out_dir, "sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}")

    print()
    print("=" * 70)
    print(f"  capacity sweep :: {args.model} x {args.task}")
    print("=" * 70)
    if "random_mean" in df.columns:
        print(f"{'p':>6} {'Top':>14} {'Random':>14} {'Gap':>8}")
        for _, r in df.iterrows():
            print(f"{r['p']:>6.3f} {r['top_mean']*100:>9.2f}±{r['top_std']*100:>3.2f} "
                  f"{r['random_mean']*100:>9.2f}±{r['random_std']*100:>3.2f} "
                  f"{(r['top_mean']-r['random_mean'])*100:>+6.2f}")
    else:
        print(f"{'p':>6} {'Top':>10}")
        for _, r in df.iterrows():
            print(f"{r['p']:>6.3f} {r['top_mean']*100:>9.2f}")


if __name__ == "__main__":
    main()
