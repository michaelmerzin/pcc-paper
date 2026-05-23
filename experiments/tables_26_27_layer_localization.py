"""Reproduce Tables 26-27: layer localization & top sensitivity neurons.

Table 26 (Phi-3-mini): top-sensitivity neuron per task
    SST-2         : L12-N2046  s=6.42   (NOT late)
    DBPedia-14    : L31-N525   s=16.34  (late)
    AG News       : L31-N525   s=14.80  (late)
    MRPC          : L31-N525   s=14.80  (late)
    GSM8K         : L29-N2812  s=15.36  (late)
    ARC-Challenge : L29-N2024  s=11.53  (late)

Table 27 (Mistral-7B): layer localization at auto-selected p*
    AG News    : top=L31-N3901   L31=23.5%  L28-31≈55%  L26-31≈72%
    DBPedia-14 : top=L31-N2070   L31≈22%    L28-31≈53%  L26-31≈70%
    SST-2      : top=L31-N3901   L31≈20%    L28-31≈50%  L26-31≈65%
    MRPC       : top=L31-N2070   L31=28.6%  L28-31≈45%  L26-31≈55%
    GSM8K      : top=L30-N3901   L31≈20%    L28-31≈51%  L26-31≈71%
    MMLU       : top=L31-N2070   L31=15.6%  L28-31=59.9%  L26-31=71.0%

This script computes SCORES per task and reports the top neuron + per-layer
concentration of the top-p* selected rows.

Usage:
    python experiments/tables_26_27_layer_localization.py --model phi3_mini --tasks sst2 dbpedia_14 ag_news mrpc gsm8k arc_challenge
    python experiments/tables_26_27_layer_localization.py --model mistral7b --tasks ag_news dbpedia_14 sst2 mrpc gsm8k mmlu
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj, get_layers
from pcc.tasks import get_task, effective_k_shots, TASKS
from pcc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import select_rows
from pcc.auto_sparsity import auto_select_rows_xdelta


def analyze_one(model, tokenizer, cfg, weights_snapshot, verbose=True):
    """Compute SCORES, run auto-p* selector, return layer-localization stats."""
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len
    use_canonical = (task.canonical_demos is not None
                     and k == len(task.canonical_demos)
                     and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, 0, use_canonical)

    # Build G_full
    _, cp_ns, _ = evaluate(model, tokenizer, task, calib_pool,
                            [[] for _ in calib_pool], cfg.max_seq_len, verbose=False)
    _, cp_fs, _ = evaluate(model, tokenizer, task, calib_pool, calib_shots,
                            cfg.max_seq_len, verbose=False)
    g_idx = build_correction_set(calib_pool, cp_ns, cp_fs, task)
    G_full = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    print(f"  |G_full| = {len(G_full)}")

    # SCORES (paper Eq. 2)
    SCORES = compute_sensitivity(model, tokenizer, task, G_full, G_demos,
                                  cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                  verbose=False)

    # Top sensitivity neuron (paper Table 26)
    flat = SCORES.view(-1)
    top_flat = flat.argmax().item()
    top_layer, top_channel = top_flat // cfg.hidden_size, top_flat % cfg.hidden_size
    top_score = float(SCORES[top_layer, top_channel])

    # Layer localization at auto-p* (paper Table 27)
    H_NS, Z_FS = collect_calibration_tensors(model, tokenizer, task, G_full, G_demos,
                                              cfg.num_layers, cfg.hidden_size,
                                              cfg.max_seq_len, verbose=False)
    W_0 = {l: get_layers(model)[l].self_attn.o_proj.weight.detach().cpu().float()
           for l in range(cfg.num_layers)}
    selected, p_star, _ = auto_select_rows_xdelta(
        H_NS, Z_FS, W_0, cfg.num_layers, cfg.hidden_size, cfg.mu,
        cumulative_target=0.90)

    # Per-layer concentration
    by_layer = Counter(l for l, _ in selected)
    total = sum(by_layer.values()) or 1
    L = cfg.num_layers
    pct_L_last       = 100 * by_layer.get(L - 1, 0) / total
    # paper reports "L28-31" (≥ L-4) and "L26-31" (≥ L-6) for Mistral (L=32)
    # use the same offsets relative to L
    pct_last4 = 100 * sum(c for l, c in by_layer.items() if l >= L - 4) / total
    pct_last6 = 100 * sum(c for l, c in by_layer.items() if l >= L - 6) / total

    return {
        "task": cfg.task_key,
        "top_neuron": f"L{top_layer}-N{top_channel}",
        "top_neuron_score": top_score,
        "top_neuron_late": top_layer >= L - 4,
        "auto_p_star": p_star,
        "n_selected": len(selected),
        "pct_top_layer": pct_L_last,
        "pct_last_4_layers": pct_last4,
        "pct_last_6_layers": pct_last6,
        "by_layer_top10": by_layer.most_common(10),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--out-dir", default="runs/tables_26_27")
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
        print(f"\n{'='*70}\n  Layer localization :: {args.model} × {tk}\n{'='*70}")
        cfg = ExperimentConfig(
            model_key=args.model, task_key=tk,
            n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
            num_layers=cfg_base.num_layers, hidden_size=cfg_base.hidden_size,
            out_dir=args.out_dir,
        )
        cfg.finalize()
        try:
            r = analyze_one(model, tokenizer, cfg, ORIG)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {tk}: {e}")
            import traceback; traceback.print_exc()
        finally:
            restore_o_proj(model, ORIG)

    with open(os.path.join(args.out_dir, f"{args.model}_localization.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Table 26: top neurons
    print(f"\n\n{'='*70}\n  TABLE 26 — Top sensitivity neurons ({args.model})\n{'='*70}")
    print(f"{'Task':<16} {'Top neuron':<14} {'s':>7} {'Late?':>6}")
    print("-" * 50)
    for r in results:
        print(f"{r['task']:<16} {r['top_neuron']:<14} {r['top_neuron_score']:>7.2f} "
              f"{'Yes' if r['top_neuron_late'] else 'No':>6}")

    # Table 27: layer localization
    print(f"\n{'='*80}\n  TABLE 27 — Layer localization at auto-p* ({args.model})\n{'='*80}")
    print(f"{'Task':<16} {'Top row':<14} {'p*':>7} {'Llast':>8} {'L last 4':>10} {'L last 6':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['task']:<16} {r['top_neuron']:<14} {r['auto_p_star']:>7.4f} "
              f"{r['pct_top_layer']:>7.1f}% {r['pct_last_4_layers']:>9.1f}% "
              f"{r['pct_last_6_layers']:>9.1f}%")


if __name__ == "__main__":
    main()
