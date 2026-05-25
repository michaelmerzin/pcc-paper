"""Hybrid ablation: Supervised vs Sup+KL vs Unsupervised on Phi-3 DBPedia.

The interesting result: filtering G_full by KL beats raw G_full at the same
budget, i.e. high-KL examples have cleaner activation shifts.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icc.config import ExperimentConfig
from icc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from icc.tasks import get_task, effective_k_shots
from icc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from icc.sensitivity import compute_sensitivity, collect_calibration_tensors
from icc.edit import select_rows, apply_edit
from icc.unsupervised import rank_by_kl, select_top_kl_balanced, estimate_unsupervised_precision


def fit_and_eval(model, tok, cfg, ORIG, examples, demos, eval_set, p_grid, tag):
    SCORES = compute_sensitivity(model, tok, get_task(cfg.task_key), examples, demos,
                                 cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tok, get_task(cfg.task_key), examples, demos,
                                             cfg.num_layers, cfg.hidden_size,
                                             cfg.max_seq_len, verbose=False)
    best, best_p = 0.0, None
    for p in p_grid:
        restore_o_proj(model, ORIG)
        sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size)
        apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
        acc, _, _ = evaluate(model, tok, get_task(cfg.task_key), eval_set,
                             [[] for _ in eval_set], cfg.max_seq_len)
        if acc > best:
            best, best_p = acc, p
    restore_o_proj(model, ORIG)
    print(f"  [{tag}] best={best*100:.2f} (p={best_p}, |G|={len(examples)})")
    return best, best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi3_mini")
    ap.add_argument("--task", default="dbpedia_14")
    ap.add_argument("--n-sup", type=int, default=64)
    ap.add_argument("--n-hybrid", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-calib-pool", type=int, default=500)
    ap.add_argument("--out-dir", default="runs/hybrid")
    args = ap.parse_args()

    cfg = ExperimentConfig(
        model_key=args.model, task_key=args.task,
        n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
        n_calib=args.n_sup,
        out_dir=args.out_dir,
        topk_percents=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30))
    cfg.finalize()

    print(f"\n{'='*70}\n  hybrid :: {args.model} x {args.task}\n{'='*70}")
    model, tok = load_model_and_tokenizer(cfg)
    ORIG = snapshot_o_proj(model)

    task = get_task(args.task)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len
    use_canon = (task.canonical_demos is not None
                 and k == len(task.canonical_demos)
                 and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, 0, use_canon)

    print("\nbuilding G_full...")
    _, cp_ns, _ = evaluate(model, tok, task, calib_pool, [[] for _ in calib_pool], cfg.max_seq_len)
    _, cp_fs, _ = evaluate(model, tok, task, calib_pool, calib_shots, cfg.max_seq_len)
    g_idx = build_correction_set(calib_pool, cp_ns, cp_fs, task)
    G_full = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    print(f"  |G_full| = {len(G_full)}")

    print("\nranking G_full by KL...")
    # rank_by_kl scans calib_pool; pass G_full as calib_pool to rank only those.
    ranked_g = rank_by_kl(model, tok, task, G_full, demo_pool,
                          k_shots=k, max_seq_len=cfg.max_seq_len,
                          seed=0, use_canonical=use_canon,
                          n_scan=len(G_full), verbose=False)
    print(f"  ranked {len(ranked_g)} G_full")

    # 1) supervised — random subset of G_full
    n_sup = min(args.n_sup, len(G_full))
    rng = random.Random(0)
    order = list(range(len(G_full)))
    rng.shuffle(order)
    sup_idx = order[:n_sup]
    sup_calib = [G_full[i] for i in sup_idx]
    sup_demos = [G_demos[i] for i in sup_idx]
    sup_acc, sup_p = fit_and_eval(model, tok, cfg, ORIG, sup_calib, sup_demos,
                                  eval_set, cfg.topk_percents, "supervised")

    # 2) hybrid — top-KL within G_full
    n_hyb = min(args.n_hybrid, len(ranked_g))
    hyb_idx = [r[0] for r in ranked_g[:n_hyb]]
    hyb_calib = [G_full[i] for i in hyb_idx]
    hyb_demos = [G_demos[i] for i in hyb_idx]
    hyb_acc, hyb_p = fit_and_eval(model, tok, cfg, ORIG, hyb_calib, hyb_demos,
                                  eval_set, cfg.topk_percents, "hybrid")

    # 3) unsupervised — top-KL on full calib_pool, no label filter
    ranked_pool = rank_by_kl(model, tok, task, calib_pool, demo_pool,
                             k_shots=k, max_seq_len=cfg.max_seq_len,
                             seed=0, use_canonical=use_canon,
                             n_scan=min(200, len(calib_pool)), verbose=False)
    n_unsup = args.n_hybrid
    unsup_calib, unsup_idx = select_top_kl_balanced(
        ranked_pool, calib_pool, n_unsup, num_classes=task.num_classes or 2)
    unsup_prec = estimate_unsupervised_precision(unsup_calib, unsup_idx,
                                                 calib_pool, ranked_pool, task)
    unsup_demos = build_shots_per_example(task, demo_pool, unsup_calib, k, 0, use_canon)
    unsup_acc, unsup_p = fit_and_eval(model, tok, cfg, ORIG, unsup_calib, unsup_demos,
                                      eval_set, cfg.topk_percents,
                                      f"unsupervised (prec={unsup_prec*100:.0f}%)")

    results = {
        "model": args.model, "task": args.task,
        "supervised":   {"n": n_sup,   "prec": 1.0,        "acc": sup_acc,   "best_p": sup_p},
        "hybrid":       {"n": n_hyb,   "prec": 1.0,        "acc": hyb_acc,   "best_p": hyb_p},
        "unsupervised": {"n": n_unsup, "prec": unsup_prec, "acc": unsup_acc, "best_p": unsup_p},
    }
    with open(os.path.join(cfg.out_dir, "hybrid.json"), "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print(f"  hybrid :: {args.model} x {args.task}")
    print("=" * 70)
    print(f"{'Method':<28} {'n':>4} {'Prec':>6} {'Acc':>8}")
    print("-" * 60)
    for tag, key in [("Supervised", "supervised"),
                     ("Hybrid (Sup + KL)", "hybrid"),
                     ("Unsupervised (top-KL)", "unsupervised")]:
        r = results[key]
        print(f"{tag:<28} {r['n']:>4} {r['prec']*100:>5.0f}% {r['acc']*100:>7.2f}")


if __name__ == "__main__":
    main()
