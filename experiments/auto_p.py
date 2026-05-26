"""No-sweep auto-sparsity experiment runner."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from icc.auto_sparsity import auto_select_rows_xdelta
from icc.config import ExperimentConfig, MODELS
from icc.edit import apply_edit
from icc.eval_engine import evaluate, build_shots_per_example, build_correction_set
from icc.model_io import load_model_and_tokenizer, snapshot_o_proj, restore_o_proj
from icc.sensitivity import collect_calibration_tensors
from icc.tasks import get_task, effective_k_shots, TASKS


TASK_DEFAULTS = {
    "ag_news":       dict(n_eval=872, n_train_pool=128, n_calib_pool=500, bs_eval=4, max_seq_len=1024),
    "dbpedia_14":    dict(n_eval=872, n_train_pool=126, n_calib_pool=560, bs_eval=4, max_seq_len=1024),
    "sst2":          dict(n_eval=872, n_train_pool=100, n_calib_pool=500, bs_eval=4, max_seq_len=512),
    "mrpc":          dict(n_eval=408, n_train_pool=100, n_calib_pool=400, bs_eval=4, max_seq_len=512),
    "gsm8k":         dict(n_eval=150, n_train_pool=128, n_calib_pool=500, bs_eval=1, max_seq_len=1024),
    "boolq":         dict(n_eval=872, n_train_pool=100, n_calib_pool=400, bs_eval=2, max_seq_len=1536),
    "arc_challenge": dict(n_eval=500, n_train_pool=120, n_calib_pool=300, bs_eval=2, max_seq_len=1024),
    "mmlu":          dict(n_eval=500, n_train_pool=285, n_calib_pool=500, bs_eval=1, max_seq_len=2048),
}


def _recovery_pct(ns_acc, fs_acc, auto_acc):
    denom = fs_acc - ns_acc
    if abs(denom) < 1e-8:
        return 0.0
    return 100.0 * (auto_acc - ns_acc) / denom


def run_one_task(model, tok, cfg, weights_snapshot, fs_seed, cumulative_target):
    task = get_task(cfg.task_key)
    k = effective_k_shots(cfg, task)
    cfg.max_seq_len = cfg.max_seq_len or task.recommended_max_seq_len

    use_canon = (
        task.canonical_demos is not None
        and k == len(task.canonical_demos)
        and cfg.use_canonical_demos
    )

    demo_pool, calib_pool, eval_set = task.load_data(cfg)
    eval_shots = build_shots_per_example(task, demo_pool, eval_set, k, fs_seed, use_canon)
    calib_shots = build_shots_per_example(task, demo_pool, calib_pool, k, fs_seed, use_canon)

    ns_acc, _, _ = evaluate(
        model, tok, task, eval_set, [[] for _ in eval_set], cfg.max_seq_len, bs=cfg.bs_eval
    )
    fs_acc, _, _ = evaluate(
        model, tok, task, eval_set, eval_shots, cfg.max_seq_len, bs=cfg.bs_eval
    )

    _, cp_ns, _ = evaluate(
        model, tok, task, calib_pool, [[] for _ in calib_pool], cfg.max_seq_len, bs=cfg.bs_eval
    )
    _, cp_fs, _ = evaluate(
        model, tok, task, calib_pool, calib_shots, cfg.max_seq_len, bs=cfg.bs_eval
    )
    g_idx = build_correction_set(calib_pool, cp_ns, cp_fs, task)
    if len(g_idx) < 4:
        raise RuntimeError(f"G_full too small for cross-fit: {len(g_idx)}")

    G = [calib_pool[i] for i in g_idx]
    G_demos = [calib_shots[i] for i in g_idx]

    H_NS, Z_FS = collect_calibration_tensors(
        model, tok, task, G, G_demos, cfg.num_layers, cfg.hidden_size, cfg.max_seq_len, verbose=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected, p_star, _ = auto_select_rows_xdelta(
        H_NS=H_NS,
        Z_FS=Z_FS,
        W0=weights_snapshot,
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        mu=cfg.mu,
        cumulative_target=cumulative_target,
        seed=fs_seed,
        device=device,
        verbose=False,
    )

    restore_o_proj(model, weights_snapshot)
    n_edited, _ = apply_edit(model, selected, H_NS, Z_FS, cfg.mu, cfg.alpha)
    auto_acc, _, _ = evaluate(
        model, tok, task, eval_set, [[] for _ in eval_set], cfg.max_seq_len, bs=cfg.bs_eval
    )
    restore_o_proj(model, weights_snapshot)

    recovery = _recovery_pct(ns_acc, fs_acc, auto_acc)
    print(
        f"{cfg.task_key:<14} NS={ns_acc*100:6.2f} "
        f"FS={fs_acc*100:6.2f} Auto={auto_acc*100:6.2f} "
        f"p*={p_star:.3f} rec={recovery:6.1f}%"
    )

    return {
        "model": cfg.model_key,
        "task": cfg.task_key,
        "ns_acc": ns_acc,
        "fs_acc": fs_acc,
        "auto_edit_acc": auto_acc,
        "recovery_pct": recovery,
        "p_star": p_star,
        "n_selected": n_edited,
        "g_size": len(G),
        "fs_seed": fs_seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--tasks", nargs="+", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--out-dir", default="runs/auto_p")
    ap.add_argument("--n-eval", type=int, default=0, help="0 -> task defaults")
    ap.add_argument("--n-calib-pool", type=int, default=0, help="0 -> task defaults")
    ap.add_argument("--fs-seed", type=int, default=1)
    ap.add_argument("--cumulative-target", type=float, default=0.90)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    first_task = args.tasks[0]
    first_defaults = TASK_DEFAULTS.get(first_task, {})
    cfg0 = ExperimentConfig(
        model_key=args.model,
        task_key=first_task,
        n_eval=args.n_eval if args.n_eval > 0 else first_defaults.get("n_eval", 150),
        n_train_pool=first_defaults.get("n_train_pool", 128),
        n_calib_pool=(
            args.n_calib_pool if args.n_calib_pool > 0 else first_defaults.get("n_calib_pool", 500)
        ),
        bs_eval=first_defaults.get("bs_eval", 1),
        max_seq_len=first_defaults.get("max_seq_len"),
        out_dir=args.out_dir,
    )
    cfg0.finalize()

    model, tok = load_model_and_tokenizer(cfg0)
    ORIG = snapshot_o_proj(model)

    rows = []
    for task_key in args.tasks:
        defaults = TASK_DEFAULTS.get(task_key, {})
        cfg = ExperimentConfig(
            model_key=args.model,
            task_key=task_key,
            n_eval=args.n_eval if args.n_eval > 0 else defaults.get("n_eval", 150),
            n_train_pool=defaults.get("n_train_pool", 128),
            n_calib_pool=(
                args.n_calib_pool if args.n_calib_pool > 0 else defaults.get("n_calib_pool", 500)
            ),
            bs_eval=defaults.get("bs_eval", 1),
            max_seq_len=defaults.get("max_seq_len"),
            num_layers=cfg0.num_layers,
            hidden_size=cfg0.hidden_size,
            out_dir=args.out_dir,
        )
        cfg.finalize()

        try:
            row = run_one_task(
                model=model,
                tok=tok,
                cfg=cfg,
                weights_snapshot=ORIG,
                fs_seed=args.fs_seed,
                cumulative_target=args.cumulative_target,
            )
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] {task_key}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            restore_o_proj(model, ORIG)

    out_path = os.path.join(args.out_dir, f"{args.model}_auto_p.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
