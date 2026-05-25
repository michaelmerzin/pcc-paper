"""PEFT/FV/ICV/PCC head-to-head on Mistral-7B. All methods see the same G_full
and the same eval set."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from pcc.config import ExperimentConfig
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import select_rows, apply_edit
from pcc.baselines.function_vector import (
    collect_full_outputs, compute_function_vector_delta, FunctionVectorInjector,
)
from pcc.baselines import peft as peft_lib
from pcc.baselines.dora import train_dora
from pcc.baselines.loreft import train_loreft, unload_loreft
from pcc.baselines.icv import run_icv_sweep


# Best-p found in the main sweep, used to skip the inner sweep when running peft compare.
MISTRAL_BEST_P = {"sst2": 0.10, "mrpc": 0.10, "dbpedia_14": 0.25, "mmlu": 0.20}

DORA_RANK = 8
DORA_LR = 2e-4
LOREFT_RANK = 4
LOREFT_LR = 5e-4
LOREFT_LAYER_SUBSET = "late_quarter"
ICV_LAMBDA_GRID = (0.05, 0.10, 0.15, 0.20)


def _gpu_peak_mb():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def setup_data(model, tok, cfg, verbose=True):
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    use_canon = (task.canonical_demos is not None
                 and k == len(task.canonical_demos)
                 and cfg.use_canonical_demos)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, 0, use_canon)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, 0, use_canon)

    if verbose:
        print("computing NS/FS on calib_pool to build G_full...")
    _, cp_ns, _ = evaluate(model, tok, task, calib_pool, [[] for _ in calib_pool],
                           cfg.max_seq_len)
    _, cp_fs, _ = evaluate(model, tok, task, calib_pool, calib_shots, cfg.max_seq_len)
    g_idx = build_correction_set(calib_pool, cp_ns, cp_fs, task)
    G_full = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    if verbose:
        print(f"|G_full| = {len(G_full)}")

    return {
        "task": task, "k": k,
        "demo_pool": demo_pool, "calib_pool": calib_pool, "calib_shots": calib_shots,
        "eval_set": eval_set, "eval_shots": eval_shots,
        "g_idx": g_idx, "G_full": G_full, "G_demos": G_demos,
        "use_canonical": use_canon,
    }


# --- method runners ---------------------------------------------------------

def method_no_shot(model, tok, cfg, data, _ws):
    acc, _, _ = evaluate(model, tok, data["task"], data["eval_set"],
                         [[] for _ in data["eval_set"]], cfg.max_seq_len)
    return {"acc": acc, "adapt_time_s": 0.0, "extra_mem_mb": 0.0, "n_params_trained": 0}


def method_few_shot(model, tok, cfg, data, _ws):
    acc, _, _ = evaluate(model, tok, data["task"], data["eval_set"],
                         data["eval_shots"], cfg.max_seq_len)
    return {"acc": acc, "adapt_time_s": 0.0, "extra_mem_mb": 0.0, "n_params_trained": 0}


def method_function_vector(model, tok, cfg, data, _ws):
    _reset_peak()
    t0 = time.time()
    Z_NS, Z_FS = collect_full_outputs(
        model, tok, data["task"], data["G_full"], data["G_demos"],
        calib_shots_ns=[[] for _ in data["G_full"]],
        num_layers=cfg.num_layers, max_seq_len=cfg.max_seq_len)
    deltas = compute_function_vector_delta(Z_NS, Z_FS)
    t_adapt = time.time() - t0
    with FunctionVectorInjector(model, deltas, selected_rows=None):
        acc, _, _ = evaluate(model, tok, data["task"], data["eval_set"],
                             [[] for _ in data["eval_set"]], cfg.max_seq_len)
    return {"acc": acc, "adapt_time_s": t_adapt, "extra_mem_mb": _gpu_peak_mb(),
            "n_params_trained": 0}


def method_icv(model, tok, cfg, data, _ws):
    _reset_peak()
    info = run_icv_sweep(
        model, tok, data["task"], data["G_full"], data["G_demos"], data["eval_set"],
        num_layers=cfg.num_layers, max_seq_len=cfg.max_seq_len,
        lambda_grid=ICV_LAMBDA_GRID, verbose=True)
    return {"acc": info["best_acc"], "adapt_time_s": info["collect_time_s"],
            "extra_mem_mb": _gpu_peak_mb(),
            "n_params_trained": 0, "best_lambda": info["best_lambda"],
            "per_lambda": {f"{k:.2f}": v for k, v in info["per_lambda"].items()}}


def method_pcc(model, tok, cfg, data, ws, sparsity):
    _reset_peak()
    t0 = time.time()
    scores = compute_sensitivity(
        model, tok, data["task"], data["G_full"], data["G_demos"],
        cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(
        model, tok, data["task"], data["G_full"], data["G_demos"],
        cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False)
    sel = select_rows(scores, sparsity, "top", cfg.num_layers, cfg.hidden_size)
    n_edit, _ = apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
    t_adapt = time.time() - t0

    acc, _, _ = evaluate(model, tok, data["task"], data["eval_set"],
                         [[] for _ in data["eval_set"]], cfg.max_seq_len)
    restore_o_proj(model, ws)
    return {"acc": acc, "adapt_time_s": t_adapt, "extra_mem_mb": _gpu_peak_mb(),
            "n_params_trained": 0, "n_edited": n_edit, "sparsity": sparsity}


def method_peft(model, tok, cfg, data, _ws, kind):
    _reset_peak()
    fn = {"lora": peft_lib.train_lora, "ia3": peft_lib.train_ia3,
          "prompt_tuning": peft_lib.train_prompt_tuning,
          "prefix_tuning": peft_lib.train_prefix_tuning}[kind]
    info = fn(model, tok, data["task"], data["G_full"], data["G_demos"],
              cfg.max_seq_len, epochs=3, verbose=False)
    pm = info.pop("peft_model")
    acc, _, _ = evaluate(pm, tok, data["task"], data["eval_set"],
                         [[] for _ in data["eval_set"]], cfg.max_seq_len)
    peft_lib.unload_peft(pm)
    return {"acc": acc, "adapt_time_s": info["wall_clock_s"], "extra_mem_mb": _gpu_peak_mb(),
            "n_params_trained": info["n_params"]}


def method_dora(model, tok, cfg, data, _ws):
    _reset_peak()
    info = train_dora(model, tok, data["task"], data["G_full"], data["G_demos"],
                      cfg.max_seq_len, rank=DORA_RANK, lr=DORA_LR, epochs=3, verbose=False)
    pm = info.pop("peft_model")
    acc, _, _ = evaluate(pm, tok, data["task"], data["eval_set"],
                         [[] for _ in data["eval_set"]], cfg.max_seq_len)
    peft_lib.unload_peft(pm)
    return {"acc": acc, "adapt_time_s": info["wall_clock_s"], "extra_mem_mb": _gpu_peak_mb(),
            "n_params_trained": info["n_params"], "rank": DORA_RANK}


def method_loreft(model, tok, cfg, data, _ws):
    _reset_peak()
    info = train_loreft(
        model, tok, data["task"], data["G_full"], data["G_demos"],
        max_seq_len=cfg.max_seq_len,
        num_layers=cfg.num_layers, hidden_size=cfg.hidden_size,
        rank=LOREFT_RANK, lr=LOREFT_LR, epochs=3,
        layer_subset=LOREFT_LAYER_SUBSET, verbose=False)
    # evaluate with hooks still attached
    acc, _, _ = evaluate(model, tok, data["task"], data["eval_set"],
                         [[] for _ in data["eval_set"]], cfg.max_seq_len)
    unload_loreft(model, info)
    return {"acc": acc, "adapt_time_s": info["wall_clock_s"], "extra_mem_mb": _gpu_peak_mb(),
            "n_params_trained": info["n_params"],
            "rank": LOREFT_RANK, "layer_subset": LOREFT_LAYER_SUBSET}


REGISTRY = {
    "no_shot": method_no_shot,
    "few_shot": method_few_shot,
    "function_vector": method_function_vector,
    "icv": method_icv,
    "lora":          lambda m, t, c, d, w: method_peft(m, t, c, d, w, "lora"),
    "ia3":           lambda m, t, c, d, w: method_peft(m, t, c, d, w, "ia3"),
    "prompt_tuning": lambda m, t, c, d, w: method_peft(m, t, c, d, w, "prompt_tuning"),
    "prefix_tuning": lambda m, t, c, d, w: method_peft(m, t, c, d, w, "prefix_tuning"),
    "dora": method_dora,
    "loreft": method_loreft,
}


def run_one_task(model, tok, cfg, methods, weights_snapshot):
    data = setup_data(model, tok, cfg)
    out = {}
    for m in methods:
        print(f"\n[{m}]...")
        restore_o_proj(model, weights_snapshot)
        if m == "pcc":
            p = MISTRAL_BEST_P[cfg.task_key]
            r = method_pcc(model, tok, cfg, data, weights_snapshot, sparsity=p)
        else:
            r = REGISTRY[m](model, tok, cfg, data, weights_snapshot)
        restore_o_proj(model, weights_snapshot)
        out[m] = r
        extra = ""
        if "best_lambda" in r:
            extra = f" best_lambda={r['best_lambda']}"
        elif "rank" in r:
            extra = f" rank={r['rank']}"
        print(f"  -> acc={r['acc']*100:.2f}  time={r['adapt_time_s']:.1f}s  "
              f"mem={r['extra_mem_mb']:.0f}MB{extra}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sst2", "mrpc", "dbpedia_14", "mmlu"])
    ap.add_argument("--methods", nargs="+",
                    default=["no_shot", "few_shot", "function_vector", "icv",
                             "lora", "dora", "ia3", "prompt_tuning",
                             "prefix_tuning", "loreft", "pcc"])
    ap.add_argument("--out-dir", default="runs/peft")
    ap.add_argument("--n-eval", type=int, default=872)
    ap.add_argument("--n-calib-pool", type=int, default=500)
    args = ap.parse_args()

    print(f"\n{'='*70}\n  peft compare\n  methods: {args.methods}\n  tasks: {args.tasks}\n{'='*70}")

    cfg_base = ExperimentConfig(
        model_key="mistral7b", task_key=args.tasks[0],
        n_eval=args.n_eval, n_calib_pool=args.n_calib_pool, out_dir=args.out_dir)
    cfg_base.finalize()
    model, tok = load_model_and_tokenizer(cfg_base)
    ORIG = snapshot_o_proj(model)

    rows = []
    for tk in args.tasks:
        cfg = ExperimentConfig(
            model_key="mistral7b", task_key=tk,
            n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
            num_layers=cfg_base.num_layers, hidden_size=cfg_base.hidden_size,
            out_dir=args.out_dir)
        cfg.finalize()
        print(f"\n{'='*70}\n  {tk}\n{'='*70}")
        results = run_one_task(model, tok, cfg, args.methods, ORIG)
        for m, r in results.items():
            rows.append({"task": tk, "method": m, **r})
        with open(os.path.join(cfg.out_dir, f"{tk}_methods.json"), "w") as f:
            json.dump(results, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "peft_compare.csv"), index=False)
    print()
    print("=" * 80)
    print("  peft compare results (Mistral-7B)")
    print("=" * 80)
    print(df.pivot(index="method", columns="task", values="acc").to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
