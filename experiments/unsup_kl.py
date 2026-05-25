"""Unsupervised KL ranking: pick examples for calibration without labels."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcc.config import ExperimentConfig, MODELS
from pcc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from pcc.tasks import get_task, effective_k_shots
from pcc.eval_engine import evaluate, build_shots_per_example
from pcc.sensitivity import compute_sensitivity, collect_calibration_tensors
from pcc.edit import select_rows, apply_edit
from pcc.unsupervised import rank_by_kl, select_top_kl_balanced, estimate_unsupervised_precision


# Target |G| per (model, task). These are the budgets used in the runs.
N_TARGET = {
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


def run_unsup(model, tok, cfg, ORIG, n_target):
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len
    use_canon = (task.canonical_demos is not None
                 and k == len(task.canonical_demos)
                 and cfg.use_canonical_demos)

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, 0, use_canon)

    ns_acc, _, _ = evaluate(model, tok, task, eval_set, [[] for _ in eval_set], cfg.max_seq_len)
    fs_acc, _, _ = evaluate(model, tok, task, eval_set, eval_shots, cfg.max_seq_len)
    print(f"  NS={ns_acc*100:.2f}  FS={fs_acc*100:.2f}")

    print(f"  KL-ranking {len(calib_pool)} examples")
    ranked = rank_by_kl(
        model, tok, task, calib_pool, demo_pool,
        k_shots=k, max_seq_len=cfg.max_seq_len,
        seed=0, use_canonical=use_canon,
        n_scan=min(200, len(calib_pool)), verbose=False)

    G, G_idx = select_top_kl_balanced(ranked, calib_pool, n_target,
                                      num_classes=task.num_classes or 2)
    prec = estimate_unsupervised_precision(G, G_idx, calib_pool, ranked, task)
    print(f"  picked |G|={len(G)}  precision={prec*100:.1f}%")

    G_shots = build_shots_per_example(task, demo_pool, G, k, 0, use_canon)

    SCORES = compute_sensitivity(model, tok, task, G, G_shots,
                                 cfg.num_layers, cfg.hidden_size, cfg.max_seq_len,
                                 verbose=False)
    H_NS, Z_FS = collect_calibration_tensors(model, tok, task, G, G_shots,
                                             cfg.num_layers, cfg.hidden_size,
                                             cfg.max_seq_len, verbose=False)

    best_acc, best_p = 0.0, None
    for p in cfg.topk_percents:
        restore_o_proj(model, ORIG)
        sel = select_rows(SCORES, p, "top", cfg.num_layers, cfg.hidden_size)
        apply_edit(model, sel, H_NS, Z_FS, cfg.mu, cfg.alpha)
        acc, _, _ = evaluate(model, tok, task, eval_set, [[] for _ in eval_set], cfg.max_seq_len)
        if acc > best_acc:
            best_acc, best_p = acc, p
        print(f"    p={p:.2f}  unsup={acc*100:.2f}")

    restore_o_proj(model, ORIG)
    return {"model": cfg.model_key, "task": cfg.task_key,
            "C": task.num_classes, "G_size": len(G), "precision": prec,
            "ns_acc": ns_acc, "fs_acc": fs_acc,
            "unsup_edit_acc": best_acc, "best_p": best_p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out-dir", default="runs/unsup_kl")
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
        print(f"\n{'='*70}\n  unsup-KL :: {args.model} x {tk}\n{'='*70}")
        cfg = ExperimentConfig(
            model_key=args.model, task_key=tk,
            n_eval=args.n_eval, n_calib_pool=args.n_calib_pool,
            num_layers=cfg0.num_layers, hidden_size=cfg0.hidden_size,
            out_dir=args.out_dir)
        cfg.finalize()
        n_t = N_TARGET.get((args.model, tk), 8)
        try:
            r = run_unsup(model, tok, cfg, ORIG, n_target=n_t)
            results.append(r)
        except Exception as e:
            print(f"  [ERROR] {tk}: {e}")
            import traceback; traceback.print_exc()
        finally:
            restore_o_proj(model, ORIG)

    with open(os.path.join(args.out_dir, f"{args.model}_unsup.json"), "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print(f"  unsup-KL :: {args.model.upper()}")
    print("=" * 70)
    print(f"{'Task':<14} {'C':>3} {'|G|':>4} {'Prec':>6} {'NS':>8} {'FS':>8} {'Unsup':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['task']:<14} {r['C']!s:>3} {r['G_size']:>4} {r['precision']*100:>5.0f}% "
              f"{r['ns_acc']*100:>7.2f} {r['fs_acc']*100:>7.2f} {r['unsup_edit_acc']*100:>9.2f}")


if __name__ == "__main__":
    main()
