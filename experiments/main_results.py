"""Main results: PCC across 3 models x 8 tasks, multi-seed."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.pipeline import run_pcc_multi_seed
from pcc.tasks import TASKS


# Per-task settings used in the runs. n_calib is *small* for SST-2/MRPC on
# purpose — that's where PCC's ridge anchor actually pulls weight against the
# gradient-based baselines.
TASK_DEFAULTS = {
    "ag_news":       dict(n_eval=872, n_train_pool=128, n_calib_pool=500, n_calib=32, bs_eval=4, max_seq_len=1024, rand_baseline_seeds=(42, 137, 271)),
    "dbpedia_14":    dict(n_eval=872, n_train_pool=126, n_calib_pool=560, n_calib=14, bs_eval=4, max_seq_len=1024, rand_baseline_seeds=(42, 137, 271)),
    "sst2":          dict(n_eval=872, n_train_pool=100, n_calib_pool=500, n_calib=4,  bs_eval=4, max_seq_len=512,  rand_baseline_seeds=(42, 137, 271)),
    "mrpc":          dict(n_eval=408, n_train_pool=100, n_calib_pool=400, n_calib=4,  bs_eval=4, max_seq_len=512,  rand_baseline_seeds=(42, 137, 271)),
    "gsm8k":         dict(n_eval=150, n_train_pool=128, n_calib_pool=500, n_calib=8,  bs_eval=1, max_seq_len=1024, rand_baseline_seeds=(42, 137, 271)),
    "boolq":         dict(n_eval=872, n_train_pool=100, n_calib_pool=400, n_calib=16, bs_eval=2, max_seq_len=1536, rand_baseline_seeds=(42, 137)),
    "arc_challenge": dict(n_eval=500, n_train_pool=120, n_calib_pool=300, n_calib=16, bs_eval=2, max_seq_len=1024, rand_baseline_seeds=(42, 137)),
    "mmlu":          dict(n_eval=500, n_train_pool=285, n_calib_pool=500, n_calib=32, bs_eval=1, max_seq_len=2048, rand_baseline_seeds=(42, 137)),
}

FS_SEEDS = {
    "ag_news":       (1, 2),
    "dbpedia_14":    (1, 2),
    "sst2":          (1, 2),
    "mrpc":          (1, 2),
    "gsm8k":         (1, 2),
    "boolq":         (1, 2),
    "arc_challenge": (1, 2),
    "mmlu":          (0,),    # single seed for mmlu — expensive and stable enough
}


def _to_json(r):
    return {
        "model_key": r["model_key"], "task_key": r["task_key"], "fs_seed": r["fs_seed"],
        "no_shot_acc": r["no_shot_acc"], "few_shot_acc": r["few_shot_acc"],
        "g_full_size": r["g_full_size"], "n_calib": r["n_calib"],
        "best_top_p": r["best_top_p"], "best_top_acc": r["best_top_acc"],
        "delta_fs": r["delta_fs"],
        "top_neuron": list(r["top_neuron"]),
        "top_neuron_score": r["top_neuron_score"],
        "top_accs": {f"{p:.2f}": v for p, v in r["top_accs"].items()},
        "random_accs": {f"{p:.2f}": v for p, v in r["random_accs"].items()},
        "t_total_s": r["t_total_s"], "t_solve_avg_s": r["t_solve_avg_s"],
    }


def run_one(model_key, task_key, out_root, n_eval, n_calib, fs_seeds, include_random):
    if task_key not in TASKS:
        raise KeyError(f"unknown task: {task_key}")

    defaults = TASK_DEFAULTS.get(task_key, {})
    use_n_eval = n_eval if n_eval > 0 else defaults.get("n_eval", 150)
    use_n_calib = n_calib if n_calib > 0 else defaults.get("n_calib", 64)
    use_fs_seeds = tuple(fs_seeds) if fs_seeds else FS_SEEDS.get(task_key, (1, 2))

    cfg = ExperimentConfig(
        model_key=model_key, task_key=task_key,
        n_eval=use_n_eval, n_calib=use_n_calib,
        n_train_pool=defaults.get("n_train_pool", 128),
        n_calib_pool=defaults.get("n_calib_pool", 500),
        bs_eval=defaults.get("bs_eval", 1),
        max_seq_len=defaults.get("max_seq_len"),
        fs_seeds=use_fs_seeds,
        rand_baseline_seeds=defaults.get("rand_baseline_seeds", (42, 137, 271)),
        out_dir=out_root,
        run_name=f"{model_key}/{task_key}",
    )
    cfg.finalize()

    print()
    print("=" * 70)
    print(f"  {model_key} x {task_key}")
    print(f"  n_eval={cfg.n_eval} n_calib={cfg.n_calib} n_calib_pool={cfg.n_calib_pool} msl={cfg.max_seq_len}")
    print(f"  fs_seeds={cfg.fs_seeds} rand_seeds={cfg.rand_baseline_seeds}")
    print("=" * 70)

    model, tok = load_model_and_tokenizer(cfg)
    ORIG = snapshot_o_proj(model)
    try:
        results = run_pcc_multi_seed(model, tok, cfg,
                                     weights_snapshot=ORIG,
                                     include_random_baseline=include_random,
                                     verbose=True)
    finally:
        restore_o_proj(model, ORIG)

    for r in results:
        with open(os.path.join(cfg.out_dir, f"seed{r['fs_seed']}.json"), "w") as f:
            json.dump(_to_json(r), f, indent=2)
        rows = []
        for p in sorted(r["top_accs"]):
            row = {"p": p, "top_acc": r["top_accs"][p]}
            if r["random_accs"]:
                row["random_mean"] = r["random_accs"][p]["mean"]
                row["random_std"] = r["random_accs"][p]["std"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(cfg.out_dir, f"seed{r['fs_seed']}_p_sweep.csv"), index=False)

    summary = {
        "model": model_key, "task": task_key, "n_seeds": len(results),
        "no_shot_mean":  float(np.mean([r["no_shot_acc"] for r in results])),
        "no_shot_std":   float(np.std([r["no_shot_acc"] for r in results])),
        "few_shot_mean": float(np.mean([r["few_shot_acc"] for r in results])),
        "few_shot_std":  float(np.std([r["few_shot_acc"] for r in results])),
        "best_pcc_acc_mean": float(np.mean([r["best_top_acc"] for r in results])),
        "best_pcc_acc_std":  float(np.std([r["best_top_acc"] for r in results])),
        "delta_fs_mean":     float(np.mean([r["delta_fs"] for r in results])),
        "best_p_seed0":      results[0]["best_top_p"],
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"summary {model_key}/{task_key}:")
    print(f"  NS  = {summary['no_shot_mean']*100:.2f} ± {summary['no_shot_std']*100:.2f}")
    print(f"  FS  = {summary['few_shot_mean']*100:.2f} ± {summary['few_shot_std']*100:.2f}")
    print(f"  PCC = {summary['best_pcc_acc_mean']*100:.2f} ± {summary['best_pcc_acc_std']*100:.2f} "
          f"(best_p={summary['best_p_seed0']})")
    print(f"  dFS = {summary['delta_fs_mean']*100:+.2f}pp")

    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--tasks", nargs="+", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--out-dir", default="runs/main")
    ap.add_argument("--n-eval", type=int, default=0,
                    help="0 -> use TASK_DEFAULTS")
    ap.add_argument("--n-calib", type=int, default=0,
                    help="0 -> use TASK_DEFAULTS")
    ap.add_argument("--fs-seeds", type=int, nargs="+", default=[],
                    help="empty -> use FS_SEEDS")
    ap.add_argument("--no-random-baseline", action="store_true")
    args = ap.parse_args()

    summaries = []
    for tk in args.tasks:
        try:
            s = run_one(args.model, tk, args.out_dir,
                        n_eval=args.n_eval, n_calib=args.n_calib,
                        fs_seeds=args.fs_seeds,
                        include_random=not args.no_random_baseline)
            summaries.append(s)
        except Exception as e:
            print(f"[ERROR] {args.model}/{tk}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)
    print(f"  {args.model.upper()}")
    print("=" * 80)
    print(f"{'Task':<16} {'NS':>8} {'FS':>10} {'Best PCC':>12} {'dFS':>8} {'best p':>8}")
    print("-" * 80)
    for s in summaries:
        ns = s["no_shot_mean"] * 100
        fs = f"{s['few_shot_mean']*100:.2f}"
        if s["few_shot_std"] > 0.01:
            fs += f"±{s['few_shot_std']*100:.2f}"
        pcc = f"{s['best_pcc_acc_mean']*100:.2f}"
        if s["best_pcc_acc_std"] > 0.01:
            pcc += f"±{s['best_pcc_acc_std']*100:.2f}"
        print(f"{s['task']:<16} {ns:>7.2f} {fs:>10} {pcc:>12} "
              f"{s['delta_fs_mean']*100:>+7.2f} {s['best_p_seed0']:>8.2f}")


if __name__ == "__main__":
    main()
