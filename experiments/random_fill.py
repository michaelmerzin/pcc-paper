"""Sanity check: replace W*_S with U[-1,1] noise. Expected to collapse to ~0%."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from icc.config import ExperimentConfig
from icc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from icc.tasks import get_task, effective_k_shots
from icc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from icc.sensitivity import compute_sensitivity, collect_calibration_tensors
from icc.edit import select_rows, apply_edit
from icc.baselines.random_fill import apply_random_fill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral7b")
    ap.add_argument("--task", default="dbpedia_14")
    ap.add_argument("--sparsities", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--fill-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fs-seed", type=int, default=1)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-calib-pool", type=int, default=500)
    ap.add_argument("--out-dir", default="runs/random_fill")
    args = ap.parse_args()

    cfg = ExperimentConfig(model_key=args.model, task_key=args.task,
                           n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
                           out_dir=args.out_dir)
    cfg.finalize()

    print(f"\n{'='*70}\n  random-fill :: {args.model} x {args.task}\n{'='*70}")
    model, tok = load_model_and_tokenizer(cfg)
    ORIG = snapshot_o_proj(model)

    task = get_task(args.task)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len
    use_canon = (task.canonical_demos is not None
                 and k == len(task.canonical_demos)
                 and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, args.fs_seed, use_canon)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, args.fs_seed, use_canon)

    print("\nbaselines:")
    ns_acc, _, _ = evaluate(model, tok, task, eval_set, [[] for _ in eval_set], cfg.max_seq_len)
    fs_acc, _, _ = evaluate(model, tok, task, eval_set, eval_shots, cfg.max_seq_len)
    print(f"  NS (seed {args.fs_seed}): {ns_acc*100:.2f}")
    print(f"  FS (seed {args.fs_seed}): {fs_acc*100:.2f}")

    _, cp_ns, _ = evaluate(model, tok, task, calib_pool, [[] for _ in calib_pool], cfg.max_seq_len)
    _, cp_fs, _ = evaluate(model, tok, task, calib_pool, calib_shots, cfg.max_seq_len)
    g_idx = build_correction_set(calib_pool, cp_ns, cp_fs, task)
    G = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    print(f"  |G_full| = {len(G)}")

    SCORES = compute_sensitivity(model, tok, task, G, G_demos,
                                 cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tok, task, G, G_demos,
                                             cfg.num_layers, cfg.hidden_size,
                                             cfg.max_seq_len, verbose=False)

    print("\nPCC closed-form:")
    pcc = {}
    for p in args.sparsities:
        restore_o_proj(model, ORIG)
        sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size)
        apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
        acc, _, _ = evaluate(model, tok, task, eval_set, [[] for _ in eval_set], cfg.max_seq_len)
        pcc[p] = acc
        print(f"  p={p:.2f}  acc={acc*100:.2f}")
    best_p = max(pcc, key=pcc.get)
    best_acc = pcc[best_p]

    print("\nrandom U[-1,1] fill (same rows):")
    rand = {}
    for p in args.sparsities:
        accs = []
        for fs in args.fill_seeds:
            restore_o_proj(model, ORIG)
            sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size)
            apply_random_fill(model, sel, seed=fs, low=-1.0, high=1.0)
            acc, _, _ = evaluate(model, tok, task, eval_set, [[] for _ in eval_set], cfg.max_seq_len)
            accs.append(acc)
        m, s = float(np.mean(accs)), float(np.std(accs))
        rand[p] = {"mean": m, "std": s, "n_seeds": len(accs)}
        print(f"  p={p:.2f}  acc={m*100:.2f}±{s*100:.2f}")

    restore_o_proj(model, ORIG)

    results = {
        "model": args.model, "task": args.task, "fs_seed": args.fs_seed,
        "no_shot_acc": ns_acc, "few_shot_acc": fs_acc,
        "pcc_at_each_p":         {f"{p:.2f}": v for p, v in pcc.items()},
        "pcc_best_p": best_p, "pcc_best_acc": best_acc,
        "random_fill_at_each_p": {f"{p:.2f}": v for p, v in rand.items()},
    }
    with open(os.path.join(cfg.out_dir, "random_fill.json"), "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print(f"  random-fill :: {args.model} x {args.task}")
    print("=" * 70)
    print(f"{'Variant':<40} {'Acc':>10} {'dNS':>10} {'dFS':>10}")
    print("-" * 70)
    print(f"{'No-shot (seed '+str(args.fs_seed)+')':<40} {ns_acc*100:>9.2f}% {'-':>10} {(ns_acc-fs_acc)*100:>+9.2f}pp")
    print(f"{'Few-shot (seed '+str(args.fs_seed)+')':<40} {fs_acc*100:>9.2f}% {(fs_acc-ns_acc)*100:>+9.2f}pp {'-':>10}")
    print(f"{'PCC closed-form p='+f'{best_p:.2f}':<40} {best_acc*100:>9.2f}% "
          f"{(best_acc-ns_acc)*100:>+9.2f}pp {(best_acc-fs_acc)*100:>+9.2f}pp")
    avg_rand = float(np.mean([v["mean"] for v in rand.values()]))
    print(f"{'Random U[-1,1] fill (avg over p)':<40} {avg_rand*100:>9.2f}% "
          f"{(avg_rand-ns_acc)*100:>+9.2f}pp {(avg_rand-fs_acc)*100:>+9.2f}pp")


if __name__ == "__main__":
    main()
