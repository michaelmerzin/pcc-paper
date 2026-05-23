"""Reproduce Table 3: no-sweep XDELTA-CAP auto-sparsity selector.

Tests the cross-fit explained-delta selector (paper §3.4) that picks p*
without any held-out evaluation, only from the activation deltas.

Paper Table 3 rows:
    Mistral DBPedia : NS=78.44  FS=83.37  Auto-Edit=84.17
    Mistral SST-2   : NS=79.01  FS=86.47  Auto-Edit=85.09
    Mistral BoolQ†  : NS=84.86  FS=84.29  Auto-Edit=84.52   (degenerate teacher)
    Phi-3   DBPedia : NS=77.75  FS=79.82  Auto-Edit=93.00
    Phi-3   MRPC    : NS=60.29  FS=77.45  Auto-Edit=79.90
    Phi-3   BoolQ   : NS=80.50  FS=86.58  Auto-Edit=82.91

Usage:
    python experiments/table3_no_sweep.py --model mistral7b --tasks dbpedia_14 sst2 boolq
    python experiments/table3_no_sweep.py --model phi3_mini --tasks dbpedia_14 mrpc boolq
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
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj, get_layers
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import apply_edit
from pcc.auto_sparsity import auto_select_rows_xdelta


def run_no_sweep(model, tokenizer, cfg: ExperimentConfig, weights_snapshot, verbose=True):
    """Run the no-sweep auto-p* pipeline for one task."""
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len

    use_canonical = (task.canonical_demos is not None
                     and k == len(task.canonical_demos)
                     and cfg.use_canonical_demos)

    # --- Splits + G_full ----------------------------------------------------
    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, 0, use_canonical)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set,  k, 0, use_canonical)

    ns_acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                              [[] for _ in eval_set], cfg.max_seq_len, verbose=False)
    fs_acc, _, _ = evaluate(model, tokenizer, task, eval_set, eval_shots,
                              cfg.max_seq_len, verbose=False)

    _, cp_ns_preds, _ = evaluate(model, tokenizer, task, calib_pool,
                                  [[] for _ in calib_pool], cfg.max_seq_len, verbose=False)
    _, cp_fs_preds, _ = evaluate(model, tokenizer, task, calib_pool, calib_shots,
                                  cfg.max_seq_len, verbose=False)
    g_idx = build_correction_set(calib_pool, cp_ns_preds, cp_fs_preds, task)
    G_full = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    if len(G_full) < 8:
        raise RuntimeError(f"G_full too small ({len(G_full)}) for cross-fit selector.")

    print(f"  splits: NS={ns_acc*100:.2f}%  FS={fs_acc*100:.2f}%  |G_full|={len(G_full)}")

    # --- Compute calibration tensors ---------------------------------------
    SCORES = compute_sensitivity(model, tokenizer, task, G_full, G_demos,
                                  cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                  verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tokenizer, task, G_full, G_demos,
                                              cfg.num_layers, cfg.hidden_size,
                                              cfg.max_seq_len, verbose=False)

    # --- Build W0 dict (full o_proj weights) for cross-fit selector --------
    W_0 = {l: get_layers(model)[l].self_attn.o_proj.weight.detach().cpu().float()
           for l in range(cfg.num_layers)}

    # --- Auto-select rows via cross-fit R² ---------------------------------
    selected_rows, p_star, R2 = auto_select_rows_xdelta(
        H_NS, Z_FS, W_0, cfg.num_layers, cfg.hidden_size, cfg.mu,
        cumulative_target=0.90,
    )
    print(f"  auto-p* = {p_star:.4f}  ({len(selected_rows)} rows selected)")

    # --- Apply edit and evaluate -------------------------------------------
    n_edited, _ = apply_edit(model, selected_rows, H_NS, Z_FS, cfg.mu, cfg.alpha)
    auto_acc, _, _ = evaluate(model, tokenizer, task, eval_set,
                                [[] for _ in eval_set], cfg.max_seq_len, verbose=False)
    restore_o_proj(model, weights_snapshot)

    recovery = (auto_acc - ns_acc) / max(1e-8, fs_acc - ns_acc)
    print(f"  auto-edit = {auto_acc*100:.2f}%  recovery = {recovery*100:.1f}%")

    return {
        "model": cfg.model_key,
        "task": cfg.task_key,
        "ns_acc": ns_acc,
        "fs_acc": fs_acc,
        "auto_edit_acc": auto_acc,
        "p_star": p_star,
        "n_edited": n_edited,
        "g_full_size": len(G_full),
        "recovery_pct": recovery * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--out-dir", default="runs/table3")
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
        print(f"\n{'='*70}\n  Table 3 :: {args.model} × {tk}\n{'='*70}")
        cfg = ExperimentConfig(
            model_key=args.model, task_key=tk,
            n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
            num_layers=cfg_base.num_layers, hidden_size=cfg_base.hidden_size,
            out_dir=args.out_dir,
        )
        cfg.finalize()
        try:
            r = run_no_sweep(model, tokenizer, cfg, ORIG, verbose=True)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {tk}: {e}")
            import traceback; traceback.print_exc()
        finally:
            restore_o_proj(model, ORIG)

    # Save and pretty-print
    with open(os.path.join(args.out_dir, f"{args.model}_table3.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*70}\n  TABLE 3 — {args.model.upper()}\n{'='*70}")
    print(f"{'Task':<16} {'NS':>8} {'FS':>8} {'Auto-Edit':>10} {'p*':>8} {'Recovery':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['task']:<16} {r['ns_acc']*100:>7.2f} {r['fs_acc']*100:>7.2f} "
              f"{r['auto_edit_acc']*100:>9.2f} {r['p_star']:>8.4f} {r['recovery_pct']:>9.1f}%")


if __name__ == "__main__":
    main()
