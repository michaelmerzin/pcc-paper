"""Reproduce Table 4: label-free KL-divergence ranker (paper §3.5).

No labels used during calibration. We rank examples by D_KL(p_ns ‖ p_fs)
over the label-token logits and pick the top |G|, balanced by predicted
few-shot label.

Paper Table 4 (Mistral-7B):
    SST-2:        C=2,  |G|=3,  Prec=100%, NS=79.01, FS=86.47, Unsup=91.17
    MRPC:         C=2,  |G|=4,  Prec=75%,  NS=75.98, FS=77.94, Unsup=80.64
    DBPedia-14:   C=14, |G|=5,  Prec=60%,  NS=78.44, FS=83.37, Unsup=90.02
    AG News:      C=4,  |G|=15, Prec=80%,  NS=79.01, FS=89.79, Unsup=80.85
    BoolQ:        C=2,  |G|=8,  Prec=75%,  NS=84.86, FS=84.06, Unsup=84.52

Paper Table 4 (Phi-3-mini):
    SST-2:        C=2,  |G|=8,  Prec=50%,  NS=69.27, FS=94.04, Unsup=93.23
    MRPC:         C=2,  |G|=16, Prec=81%,  NS=60.29, FS=75.49, Unsup=76.72
    DBPedia-14:   C=14, |G|=32, Prec=75%,  NS=77.75, FS=79.82, Unsup=98.17
    BoolQ:        C=2,  |G|=6,  Prec=67%,  NS=80.50, FS=86.58, Unsup=85.78

Usage:
    python experiments/table4_unsupervised_kl.py --model mistral7b --tasks sst2 mrpc dbpedia_14
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pcc.config import ExperimentConfig, MODELS, DEFAULT_TOPK_PERCENTS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import select_rows, apply_edit
from pcc.unsupervised import rank_by_kl, select_top_kl_balanced, estimate_unsupervised_precision


# Paper Table 4: target |G| per task
PAPER_N_TARGET = {
    ("mistral7b", "sst2"): 3,
    ("mistral7b", "mrpc"): 4,
    ("mistral7b", "dbpedia_14"): 5,
    ("mistral7b", "ag_news"): 15,
    ("mistral7b", "boolq"): 8,
    ("phi3_mini", "sst2"): 8,
    ("phi3_mini", "mrpc"): 16,
    ("phi3_mini", "dbpedia_14"): 32,
    ("phi3_mini", "boolq"): 6,
}


def run_unsupervised(model, tokenizer, cfg: ExperimentConfig, weights_snapshot,
                      n_target: int, verbose=True):
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len

    use_canonical = (task.canonical_demos is not None
                     and k == len(task.canonical_demos)
                     and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, 0, use_canonical)

    # NS / FS baselines on eval set
    ns_acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                              [[] for _ in eval_set], cfg.max_seq_len, verbose=False)
    fs_acc, _, _ = evaluate(model, tokenizer, task, eval_set, eval_shots,
                              cfg.max_seq_len, verbose=False)
    print(f"  NS={ns_acc*100:.2f}%  FS={fs_acc*100:.2f}%")

    # --- KL ranking on calib_pool (no labels!) -----------------------------
    print(f"  ranking {len(calib_pool)} examples by KL...")
    ranked = rank_by_kl(
        model, tokenizer, task, calib_pool, demo_pool,
        k_shots=k, max_seq_len=cfg.max_seq_len,
        seed=0, use_canonical=use_canonical,
        n_scan=min(200, len(calib_pool)), verbose=False,
    )

    # Pick top-KL balanced by predicted label
    G_unsup, G_idx = select_top_kl_balanced(ranked, calib_pool, n_target,
                                              num_classes=task.num_classes or 2)
    prec = estimate_unsupervised_precision(G_unsup, G_idx, calib_pool, ranked, task)
    print(f"  selected |G|={len(G_unsup)}  precision={prec*100:.1f}%")

    # Build matching shot lists for the selected calibration examples
    G_shots = build_shots_per_example(task, demo_pool, G_unsup, k, 0, use_canonical)

    # --- Compute SCORES + H_NS, Z_FS on the unsupervised G -----------------
    SCORES = compute_sensitivity(model, tokenizer, task, G_unsup, G_shots,
                                  cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                  verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tokenizer, task, G_unsup, G_shots,
                                              cfg.num_layers, cfg.hidden_size,
                                              cfg.max_seq_len, verbose=False)

    # --- Sweep over p, pick best -------------------------------------------
    best_acc, best_p = 0.0, None
    for p in cfg.topk_percents:
        restore_o_proj(model, weights_snapshot)
        sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size)
        apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
        acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                              [[] for _ in eval_set], cfg.max_seq_len, verbose=False)
        if acc > best_acc:
            best_acc, best_p = acc, p
        if verbose:
            print(f"    p={p:.2f}  unsup-edit={acc*100:.2f}%")

    restore_o_proj(model, weights_snapshot)
    return {
        "model": cfg.model_key, "task": cfg.task_key,
        "C": task.num_classes, "G_size": len(G_unsup), "precision": prec,
        "ns_acc": ns_acc, "fs_acc": fs_acc,
        "unsup_edit_acc": best_acc, "best_p": best_p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--out-dir", default="runs/table4")
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-calib-pool", type=int, default=500)
    args = parser.parse_args()

    cfg_base = ExperimentConfig(model_key=args.model, task_key=args.tasks[0],
                                  n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
                                  out_dir=args.out_dir)
    cfg_base.finalize()
    model, tokenizer = load_model_and_tokenizer(cfg_base)
    ORIG = snapshot_o_proj(model)

    results = []
    for tk in args.tasks:
        print(f"\n{'='*70}\n  Table 4 :: {args.model} × {tk}\n{'='*70}")
        cfg = ExperimentConfig(
            model_key=args.model, task_key=tk,
            n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
            num_layers=cfg_base.num_layers, hidden_size=cfg_base.hidden_size,
            out_dir=args.out_dir,
        )
        cfg.finalize()
        n_target = PAPER_N_TARGET.get((args.model, tk), 8)
        try:
            r = run_unsupervised(model, tokenizer, cfg, ORIG, n_target=n_target)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {tk}: {e}")
            import traceback; traceback.print_exc()
        finally:
            restore_o_proj(model, ORIG)

    with open(os.path.join(args.out_dir, f"{args.model}_table4.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*80}\n  TABLE 4 — {args.model.upper()}\n{'='*80}")
    print(f"{'Task':<14} {'C':>3} {'|G|':>4} {'Prec':>6} {'NS':>8} {'FS':>8} {'Unsup':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['task']:<14} {r['C']!s:>3} {r['G_size']:>4} {r['precision']*100:>5.0f}% "
              f"{r['ns_acc']*100:>7.2f} {r['fs_acc']*100:>7.2f} {r['unsup_edit_acc']*100:>9.2f}")


if __name__ == "__main__":
    main()
