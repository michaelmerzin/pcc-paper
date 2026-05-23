"""Reproduce Table 2 of the paper: PEFT baselines vs PCC on Mistral-7B.

Methods compared, all trained on the SAME correction set G_full:
  - No-shot baseline           (paper: 79.01 / 76.23 / 78.44 / 60.60)
  - Few-shot teacher           (paper: 86.47 / 77.94 / 83.37 / 59.60)
  - Function Vector            (paper: 81.77 / 78.92 / 78.56 / 60.60)
  - LoRA r=8                   (paper: 94.72 / 34.31 / 41.17 / 55.40)
  - IA3                        (paper: 92.32 / 70.83 / 87.16 / 60.40)
  - Prompt Tuning              (paper: 77.29 / 74.26 / 84.52 / 50.20)
  - Prefix Tuning              (paper:  4.82 /  0.00 /  5.05 / 21.80)
  - PCC (Ours)                 (paper: 88.65 / 80.88 / 90.71 / 61.20)

Tasks: SST-2, MRPC, DBPedia-14, MMLU.

Usage:
    python experiments/table2_peft_comparison.py --tasks sst2 mrpc dbpedia_14 mmlu \\
        --methods no_shot few_shot function_vector lora ia3 prompt_tuning prefix_tuning pcc
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd

from pcc.config import ExperimentConfig
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import select_rows, apply_edit
from pcc.baselines.function_vector import (
    collect_full_outputs, compute_function_vector_delta, FunctionVectorInjector,
)
from pcc.baselines import peft as peft_baselines


# Paper-reported best PCC sparsity per task (Mistral-7B):
MISTRAL_BEST_P = {"sst2": 0.10, "mrpc": 0.10, "dbpedia_14": 0.25, "mmlu": 0.20}


def _record_gpu_peak_mb():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def setup_data(model, tokenizer, cfg, verbose=True):
    """Build splits, G_full, and CALIB.  Shared across all methods."""
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    use_canonical = (task.canonical_demos is not None
                     and k == len(task.canonical_demos)
                     and cfg.use_canonical_demos)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, 0, use_canonical)
    eval_shots  = build_shots_per_example(task, demo_pool, eval_set,  k, 0, use_canonical)

    if verbose: print(f"[setup] computing baselines on calib_pool...")
    _, cp_ns_preds, _ = evaluate(model, tokenizer, task, calib_pool,
                                  [[] for _ in calib_pool], cfg.max_seq_len, verbose=False)
    _, cp_fs_preds, _ = evaluate(model, tokenizer, task, calib_pool,
                                  calib_shots, cfg.max_seq_len, verbose=False)
    g_idx = build_correction_set(calib_pool, cp_ns_preds, cp_fs_preds, task)
    G_full = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    if verbose: print(f"[setup] |G_full|={len(G_full)}")

    return {
        "task": task,
        "k": k,
        "demo_pool": demo_pool,
        "calib_pool": calib_pool,
        "calib_shots": calib_shots,
        "eval_set": eval_set,
        "eval_shots": eval_shots,
        "g_idx": g_idx,
        "G_full": G_full,
        "G_demos": G_demos,
        "use_canonical": use_canonical,
    }


def method_no_shot(model, tokenizer, cfg, data):
    acc, _, _ = evaluate(model, tokenizer, data["task"], data["eval_set"],
                          [[] for _ in data["eval_set"]], cfg.max_seq_len, verbose=False)
    return {"acc": acc, "adapt_time_s": 0.0, "extra_mem_mb": 0.0}


def method_few_shot(model, tokenizer, cfg, data):
    acc, _, _ = evaluate(model, tokenizer, data["task"], data["eval_set"],
                          data["eval_shots"], cfg.max_seq_len, verbose=False)
    return {"acc": acc, "adapt_time_s": 0.0, "extra_mem_mb": 0.0}


def method_function_vector(model, tokenizer, cfg, data, weights_snapshot):
    _reset_gpu_peak()
    t0 = time.time()
    Z_NS, Z_FS = collect_full_outputs(
        model, tokenizer, data["task"],
        data["G_full"], data["G_demos"],
        calib_shots_ns=[[] for _ in data["G_full"]],
        num_layers=cfg.num_layers, max_seq_len=cfg.max_seq_len,
    )
    deltas = compute_function_vector_delta(Z_NS, Z_FS)
    t_adapt = time.time() - t0

    with FunctionVectorInjector(model, deltas, selected_rows=None):
        acc, _, _ = evaluate(model, tokenizer, data["task"], data["eval_set"],
                              [[] for _ in data["eval_set"]], cfg.max_seq_len, verbose=False)
    mem = _record_gpu_peak_mb()
    return {"acc": acc, "adapt_time_s": t_adapt, "extra_mem_mb": mem}


def method_pcc(model, tokenizer, cfg, data, weights_snapshot, sparsity: float):
    _reset_gpu_peak()
    t0 = time.time()
    SCORES = compute_sensitivity(
        model, tokenizer, data["task"], data["G_full"], data["G_demos"],
        cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False,
    )
    H_NS, Z_FS = collect_calibration_tensors(
        model, tokenizer, data["task"], data["G_full"], data["G_demos"],
        cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False,
    )
    sel = select_rows(SCORES, sparsity, "top", cfg.num_layers, cfg.hidden_size)
    n_edited, _ = apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
    t_adapt = time.time() - t0

    acc, _, _ = evaluate(model, tokenizer, data["task"], data["eval_set"],
                          [[] for _ in data["eval_set"]], cfg.max_seq_len, verbose=False)
    mem = _record_gpu_peak_mb()
    restore_o_proj(model, weights_snapshot)
    return {"acc": acc, "adapt_time_s": t_adapt, "extra_mem_mb": mem,
            "n_edited": n_edited, "sparsity": sparsity}


def method_peft(model, tokenizer, cfg, data, peft_kind: str):
    _reset_gpu_peak()
    train_fn = {
        "lora":          peft_baselines.train_lora,
        "ia3":           peft_baselines.train_ia3,
        "prompt_tuning": peft_baselines.train_prompt_tuning,
        "prefix_tuning": peft_baselines.train_prefix_tuning,
    }[peft_kind]
    info = train_fn(model, tokenizer, data["task"], data["G_full"], data["G_demos"],
                    cfg.max_seq_len, epochs=3, verbose=False)
    peft_model = info.pop("peft_model")
    acc, _, _ = evaluate(peft_model, tokenizer, data["task"], data["eval_set"],
                          [[] for _ in data["eval_set"]], cfg.max_seq_len, verbose=False)
    mem = _record_gpu_peak_mb()
    # Unload adapter / merge back
    peft_baselines.unload_peft(peft_model)
    return {"acc": acc, "adapt_time_s": info["wall_clock_s"], "extra_mem_mb": mem,
            "n_params_trained": info["n_params"]}


METHOD_REGISTRY = {
    "no_shot":         method_no_shot,
    "few_shot":        method_few_shot,
    "function_vector": method_function_vector,
    "lora":            lambda m, t, c, d, _w: method_peft(m, t, c, d, "lora"),
    "ia3":             lambda m, t, c, d, _w: method_peft(m, t, c, d, "ia3"),
    "prompt_tuning":   lambda m, t, c, d, _w: method_peft(m, t, c, d, "prompt_tuning"),
    "prefix_tuning":   lambda m, t, c, d, _w: method_peft(m, t, c, d, "prefix_tuning"),
}


def run_one_task(model, tokenizer, cfg, methods, weights_snapshot):
    data = setup_data(model, tokenizer, cfg, verbose=True)
    results = {}
    for m in methods:
        print(f"\n  [{m}]...")
        restore_o_proj(model, weights_snapshot)
        if m == "pcc":
            p = MISTRAL_BEST_P[cfg.task_key]
            r = method_pcc(model, tokenizer, cfg, data, weights_snapshot, sparsity=p)
        elif m in ("no_shot", "few_shot", "function_vector"):
            r = METHOD_REGISTRY[m](model, tokenizer, cfg, data, weights_snapshot)
        else:
            r = METHOD_REGISTRY[m](model, tokenizer, cfg, data, weights_snapshot)
        restore_o_proj(model, weights_snapshot)
        results[m] = r
        print(f"  → acc={r['acc']*100:.2f}%  time={r['adapt_time_s']:.1f}s  mem={r['extra_mem_mb']:.0f}MB")
    return results


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 2 of the PCC paper.")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mrpc", "dbpedia_14", "mmlu"])
    parser.add_argument("--methods", nargs="+",
                        default=["no_shot", "few_shot", "function_vector",
                                 "lora", "ia3", "prompt_tuning", "prefix_tuning", "pcc"])
    parser.add_argument("--out-dir", default="runs/table2")
    parser.add_argument("--n-eval", type=int, default=150)
    parser.add_argument("--n-calib-pool", type=int, default=500)
    args = parser.parse_args()

    cfg_base = ExperimentConfig(
        model_key="mistral7b",
        task_key=args.tasks[0],
        n_eval=args.n_eval,
        n_calib_pool=args.n_calib_pool,
        out_dir=args.out_dir,
    )
    cfg_base.finalize()
    model, tokenizer = load_model_and_tokenizer(cfg_base)
    ORIG = snapshot_o_proj(model)

    all_rows = []
    for tk in args.tasks:
        cfg = ExperimentConfig(
            model_key="mistral7b", task_key=tk,
            n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
            num_layers=cfg_base.num_layers, hidden_size=cfg_base.hidden_size,
            out_dir=args.out_dir,
        )
        cfg.finalize()
        print(f"\n{'='*70}\n  Table 2 :: {tk}\n{'='*70}")
        results = run_one_task(model, tokenizer, cfg, args.methods, ORIG)
        for method, r in results.items():
            all_rows.append({"task": tk, "method": method, **r})

        with open(os.path.join(cfg.out_dir, f"{tk}_methods.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Pretty-print final Table 2
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out_dir, "table2_full.csv"), index=False)

    print(f"\n\n{'='*80}\n  Table 2 (Mistral-7B) — final\n{'='*80}")
    print(df.pivot(index="method", columns="task", values="acc").to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
