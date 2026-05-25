"""Layer + neuron localization. Reports:
  - top-K most-sensitive (layer, neuron) pairs
  - share of the auto-selected mass in each layer (where does the edit live?)
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj, get_layers
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.auto_sparsity import auto_select_rows_xdelta


def analyze_one(model, tok, cfg, ORIG, top_k=10):
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len
    use_canon = (task.canonical_demos is not None
                 and k == len(task.canonical_demos)
                 and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, 0, use_canon)

    print("building G_full...")
    _, cp_ns, _ = evaluate(model, tok, task, calib_pool, [[] for _ in calib_pool], cfg.max_seq_len)
    _, cp_fs, _ = evaluate(model, tok, task, calib_pool, calib_shots, cfg.max_seq_len)
    g_idx = build_correction_set(calib_pool, cp_ns, cp_fs, task)
    G = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    if len(G) < 8:
        raise RuntimeError(f"G too small ({len(G)})")

    SCORES = compute_sensitivity(model, tok, task, G, G_demos,
                                 cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tok, task, G, G_demos,
                                             cfg.num_layers, cfg.hidden_size,
                                             cfg.max_seq_len, verbose=False)
    W0 = {l: get_layers(model)[l].self_attn.o_proj.weight.detach().cpu().float()
          for l in range(cfg.num_layers)}

    # top-K most-sensitive (layer, neuron) by sensitivity score
    flat = SCORES.view(-1)
    top_idx = flat.argsort(descending=True)[:top_k].tolist()
    top_neurons = []
    for idx in top_idx:
        L, N = idx // cfg.hidden_size, idx % cfg.hidden_size
        top_neurons.append({"layer": L, "neuron": N, "sensitivity": float(flat[idx])})

    # layer distribution of auto-selected rows
    selected, p_star, _ = auto_select_rows_xdelta(
        H_NS, Z_FS, W0, cfg.num_layers, cfg.hidden_size, cfg.mu, cumulative_target=0.90)
    layer_hist = Counter(l for l, _ in selected)
    total_sel = sum(layer_hist.values())
    layer_dist = {l: {"count": c, "share": c / total_sel}
                  for l, c in sorted(layer_hist.items())}

    restore_o_proj(model, ORIG)
    return {
        "model": cfg.model_key, "task": cfg.task_key,
        "G_size": len(G), "p_star_auto": p_star, "n_selected_auto": total_sel,
        "top_neurons": top_neurons, "layer_distribution_auto": layer_dist,
        "num_layers": cfg.num_layers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out-dir", default="runs/localization")
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-calib-pool", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    cfg0 = ExperimentConfig(model_key=args.model, task_key=args.tasks[0],
                            n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
                            out_dir=args.out_dir)
    cfg0.finalize()
    model, tok = load_model_and_tokenizer(cfg0)
    ORIG = snapshot_o_proj(model)

    results = []
    for tk in args.tasks:
        print(f"\n{'='*70}\n  localization :: {args.model} x {tk}\n{'='*70}")
        cfg = ExperimentConfig(model_key=args.model, task_key=tk,
                               n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
                               num_layers=cfg0.num_layers, hidden_size=cfg0.hidden_size,
                               out_dir=args.out_dir)
        cfg.finalize()
        try:
            r = analyze_one(model, tok, cfg, ORIG, top_k=args.top_k)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {tk}: {e}")
            import traceback; traceback.print_exc()
        finally:
            restore_o_proj(model, ORIG)

    with open(os.path.join(args.out_dir, f"{args.model}_localization.json"), "w") as f:
        json.dump(results, f, indent=2)

    # top-neuron table
    print()
    print("=" * 70)
    print(f"  top {args.top_k} most-sensitive neurons :: {args.model.upper()}")
    print("=" * 70)
    for r in results:
        print(f"\n{r['task']}:")
        print(f"  {'rank':>4} {'layer':>6} {'neuron':>8} {'sensitivity':>14}")
        for i, n in enumerate(r["top_neurons"]):
            print(f"  {i+1:>4} {n['layer']:>6} {n['neuron']:>8} {n['sensitivity']:>14.6f}")

    # layer concentration
    print()
    print("=" * 70)
    print(f"  layer concentration of auto-selected rows :: {args.model.upper()}")
    print("=" * 70)
    for r in results:
        print(f"\n{r['task']}: p*={r['p_star_auto']:.4f}, n_selected={r['n_selected_auto']}")
        # top 5 layers by share
        ranked = sorted(r["layer_distribution_auto"].items(),
                        key=lambda x: x[1]["share"], reverse=True)[:5]
        print(f"  {'layer':>6} {'count':>8} {'share':>10}")
        for lyr, info in ranked:
            print(f"  {lyr:>6} {info['count']:>8} {info['share']*100:>9.2f}%")


if __name__ == "__main__":
    main()
