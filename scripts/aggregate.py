"""Dump LaTeX fragments from the runs/ directory.

Outputs:
  - main.tex          (NS / FS / PCC + dFS per task per model)
  - peft.tex          (peft compare on mistral)
  - auto_p.tex        (cross-fit auto-p recovery)
  - unsup_kl.tex      (unsupervised KL)
  - random_fill.tex   (sanity check)
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def _safe_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  skip {path}: {e}")
        return None


def write_main(runs_dir, out_dir):
    main_dir = os.path.join(runs_dir, "main")
    rows = defaultdict(dict)  # rows[task][model] = summary
    for model_dir in sorted(glob.glob(os.path.join(main_dir, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir)
        for task_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            task = os.path.basename(task_dir)
            s = _safe_json(os.path.join(task_dir, "summary.json"))
            if s:
                rows[task][model] = s

    if not rows:
        print("  main: nothing to write")
        return

    lines = [
        r"\begin{tabular}{l" + "rrr" * 3 + "}",
        r"\toprule",
        r"Task & " + " & ".join(
            [r"\multicolumn{3}{c}{" + m + "}" for m in ("Mistral-7B", "Phi-3-mini", "Gemma-2-2B")]
        ) + r" \\",
        r"     & NS & FS & PCC" * 3 + r" \\",
        r"\midrule",
    ]
    for task in sorted(rows):
        cells = [task]
        for model_key in ("mistral7b", "phi3_mini", "gemma2_2b"):
            if model_key in rows[task]:
                s = rows[task][model_key]
                cells.extend([f"{s['no_shot_mean']*100:.1f}",
                              f"{s['few_shot_mean']*100:.1f}",
                              f"{s['best_pcc_acc_mean']*100:.1f}"])
            else:
                cells.extend(["-", "-", "-"])
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    p = os.path.join(out_dir, "main.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {p}")


def write_peft(runs_dir, out_dir):
    csv = os.path.join(runs_dir, "peft", "peft_compare.csv")
    if not os.path.exists(csv):
        print("  peft: no peft_compare.csv")
        return
    import pandas as pd
    df = pd.read_csv(csv)
    piv = df.pivot(index="method", columns="task", values="acc")

    tasks = list(piv.columns)
    lines = [
        r"\begin{tabular}{l" + "r" * len(tasks) + "}",
        r"\toprule",
        "Method & " + " & ".join(tasks) + r" \\",
        r"\midrule",
    ]
    for m in piv.index:
        cells = [m]
        for t in tasks:
            v = piv.loc[m, t]
            cells.append(f"{v*100:.1f}" if v == v else "-")  # nan check
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    p = os.path.join(out_dir, "peft.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {p}")


def write_auto_p(runs_dir, out_dir):
    files = glob.glob(os.path.join(runs_dir, "auto_p", "*_auto_p.json"))
    if not files:
        print("  auto_p: nothing to write")
        return

    rows = defaultdict(dict)
    for fp in files:
        with open(fp) as f:
            arr = json.load(f)
        for r in arr:
            rows[r["task"]][r["model"]] = r

    lines = [
        r"\begin{tabular}{l" + "rrrr" * 3 + "}",
        r"\toprule",
        "Task & " + " & ".join(
            r"\multicolumn{4}{c}{" + m + "}" for m in ("Mistral-7B", "Phi-3-mini", "Gemma-2-2B")
        ) + r" \\",
        r"     & NS & FS & Auto & Rec" * 3 + r" \\",
        r"\midrule",
    ]
    for task in sorted(rows):
        cells = [task]
        for mk in ("mistral7b", "phi3_mini", "gemma2_2b"):
            if mk in rows[task]:
                r = rows[task][mk]
                cells.extend([f"{r['ns_acc']*100:.1f}", f"{r['fs_acc']*100:.1f}",
                              f"{r['auto_edit_acc']*100:.1f}",
                              f"{r['recovery_pct']:.0f}\\%"])
            else:
                cells.extend(["-", "-", "-", "-"])
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    p = os.path.join(out_dir, "auto_p.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {p}")


def write_unsup_kl(runs_dir, out_dir):
    files = glob.glob(os.path.join(runs_dir, "unsup_kl", "*_unsup.json"))
    if not files:
        print("  unsup_kl: nothing to write")
        return

    rows = []
    for fp in files:
        with open(fp) as f:
            arr = json.load(f)
        rows.extend(arr)

    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Task & $C$ & $|G|$ & Prec. & FS & Unsup-edit \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['model']} & {r['task']} & {r['C']} & {r['G_size']} & "
            f"{r['precision']*100:.0f}\\% & {r['fs_acc']*100:.1f} & "
            f"{r['unsup_edit_acc']*100:.1f}" + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    p = os.path.join(out_dir, "unsup_kl.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {p}")


def write_random_fill(runs_dir, out_dir):
    rf = _safe_json(os.path.join(runs_dir, "random_fill", "random_fill.json"))
    if not rf:
        print("  random_fill: nothing")
        return

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Variant & Acc & $\Delta$NS & $\Delta$FS \\",
        r"\midrule",
    ]
    ns = rf["no_shot_acc"]
    fs = rf["few_shot_acc"]
    pcc_p = rf["pcc_best_p"]
    pcc_acc = rf["pcc_best_acc"]
    rand = list(rf["random_fill_at_each_p"].values())
    rand_mean = sum(r["mean"] for r in rand) / len(rand)

    lines.append(f"No-shot & {ns*100:.1f} & -- & {(ns - fs)*100:+.1f} \\\\")
    lines.append(f"Few-shot & {fs*100:.1f} & {(fs - ns)*100:+.1f} & -- \\\\")
    lines.append(
        f"PCC closed-form (best $p={pcc_p:.2f}$) & {pcc_acc*100:.1f} & "
        f"{(pcc_acc - ns)*100:+.1f} & {(pcc_acc - fs)*100:+.1f} \\\\")
    lines.append(
        f"Random $U[-1,1]$ (avg over $p$) & {rand_mean*100:.1f} & "
        f"{(rand_mean - ns)*100:+.1f} & {(rand_mean - fs)*100:+.1f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    p = os.path.join(out_dir, "random_fill.tex")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="runs/latex_fragments")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("main:")
    write_main(args.runs_dir, args.out)
    print("peft:")
    write_peft(args.runs_dir, args.out)
    print("auto_p:")
    write_auto_p(args.runs_dir, args.out)
    print("unsup_kl:")
    write_unsup_kl(args.runs_dir, args.out)
    print("random_fill:")
    write_random_fill(args.runs_dir, args.out)


if __name__ == "__main__":
    main()
