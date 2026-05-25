"""No-sweep auto-sparsity: pick p* from cross-fit R^2 instead of evaluating
the whole sweep on a held-out set.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj, get_layers
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import apply_edit
from pcc.auto_sparsity import auto_select_rows_xdelta


def run_auto_p(model, tok, cfg, ORIG, verbose=True):
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len
    use_canon = (task.canonical_demos is not None
                 and k == len(task.canonical_demos)
                 and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, 0, use_canon)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, 0, use_canon)

    ns_acc, _, _ = evaluate(model, tok, task, eval_set,
                            [[] for _ in eval_set], cfg.max_seq_len)
    fs_acc, _, _ = evaluate(model, tok, task, eval_set, eval_shots, cfg.max_seq_len)

    _, cp_ns, _ = evaluate(model, tok, task, calib_pool,
                           [[] for _ in calib_pool], cfg.max_seq_len)
    _, cp_fs, _ = evaluate(model, tok, task, calib_pool, calib_shots, cfg.max_seq_len)
    g_idx = build_correction_set(calib_pool, cp_ns, cp_fs, task)
    G = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]
    if len(G) < 8:
        raise RuntimeError(f"G too small ({len(G)}) for cross-fit")

    print(f"  NS={ns_acc*100:.2f}  FS={fs_acc*100:.2f}  |G|={len(G)}")

    SCORES = compute_sensitivity(model, tok, task, G, G_demos,
                                 cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                 verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tok, task, G, G_demos,
                                             cfg.num_layers, cfg.hidden_size,
                                             cfg.max_seq_len, verbose=False)
    W0 = {l: get_layers(model)[l].self_attn.o_proj.weight.detach().cpu().float()
          for l in range(cfg.num_layers)}

    selected, p_star, R2 = auto_select_rows_xdelta(
        H_NS, Z_FS, W0, cfg.num_layers, cfg.hidden_size, cfg.mu,
        cumulative_target=0.90)
    print(f"  auto p* = {p_star:.4f}  ({len(selected)} rows)")

    n_edit, _ = apply_edit(model, selected, H_NS, Z_FS, cfg.mu, cfg.alpha)
    auto_acc, _, _ = evaluate(model, tok, task, eval_set,
                              [[] for _ in eval_set], cfg.max_seq_len)
    restore_o_proj(model, ORIG)

    recovery = (auto_acc - ns_acc) / max(1e-8, fs_acc - ns_acc)
    print(f"  auto-edit = {auto_acc*100:.2f}  recovery = {recovery*100:.1f}%")

    return {
        "model": cfg.model_key, "task": cfg.task_key,
        "ns_acc": ns_acc, "fs_acc": fs_acc,
        "auto_edit_acc": auto_acc, "p_star": p_star,
        "n_edited": n_edit, "g_full_size": len(G),
        "recovery_pct": recovery * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out-dir", default="runs/auto_p")
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-calib-pool", type=int, default=500)
    args = ap.parse_args()

    cfg0 = ExperimentConfig(model_key=args.model, task_key=args.tasks[0],
                            n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
                            out_dir=args.out_dir)
    cfg0.finalize()
    model, tok = load_model_and_tokenizer(cfg0)
    ORIG = snapshot_o_proj(model)

    results = []
    for tk in args.tasks:
        print(f"\n{'='*70}\n  auto-p :: {args.model} x {tk}\n{'='*70}")
        cfg = ExperimentConfig(model_key=args.model, task_key=tk,
                               n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
                               num_layers=cfg0.num_layers, hidden_size=cfg0.hidden_size,
                               out_dir=args.out_dir)
        cfg.finalize()
        try:
            r = run_auto_p(model, tok, cfg, ORIG)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {tk}: {e}")
            import traceback; traceback.print_exc()
        finally:
            restore_o_proj(model, ORIG)

    with open(os.path.join(args.out_dir, f"{args.model}_auto_p.json"), "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print(f"  auto-p :: {args.model.upper()}")
    print("=" * 70)
    print(f"{'Task':<16} {'NS':>8} {'FS':>8} {'Auto':>10} {'p*':>8} {'Rec':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['task']:<16} {r['ns_acc']*100:>7.2f} {r['fs_acc']*100:>7.2f} "
              f"{r['auto_edit_acc']*100:>9.2f} {r['p_star']:>8.4f} {r['recovery_pct']:>9.1f}%")


if __name__ == "__main__":
    main()
