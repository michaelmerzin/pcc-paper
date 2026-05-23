"""Aggregate JSON outputs from `experiments/` into LaTeX table fragments.

Reads `runs/table*/.../summary.json` etc. and writes a single .tex file
containing the booktabs-style fragments that drop into the paper.

Usage:
    python scripts/aggregate_results.py --runs-dir runs --out-file runs/paper_tables.tex
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from typing import List, Dict


def load_json_safely(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect_table1(runs_dir: str) -> Dict[str, List[Dict]]:
    """Read runs/table1/<model>/<task>/summary.json files."""
    out: Dict[str, List[Dict]] = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "table1", "*", "*", "summary.json"))):
        s = load_json_safely(path)
        if not s:
            continue
        out.setdefault(s["model"], []).append(s)
    return out


def fmt_table1_latex(by_model: Dict[str, List[Dict]]) -> str:
    """Render Table 1 from the per-cell summaries."""
    label_map = {
        "mistral7b": "Mistral-7B",
        "phi3_mini": "Phi-3-mini",
        "gemma2_2b": "Gemma-2-2B",
    }
    lines = [
        "\\begin{table*}[!t]",
        "\\centering\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Task & No-Shot & Few-Shot & \\textbf{Best PCC Edit} & $\\Delta$FS & Best $p$ \\\\",
        "\\midrule",
    ]
    for mk, rows in by_model.items():
        n_rows = len(rows)
        for i, r in enumerate(rows):
            prefix = f"\\multirow{{{n_rows}}}{{*}}{{{label_map.get(mk, mk)}}}" if i == 0 else ""
            ns_str = f"{r['no_shot_mean']*100:.2f}"
            fs_str = f"{r['few_shot_mean']*100:.2f}"
            if r['few_shot_std'] > 0.01:
                fs_str = f"${r['few_shot_mean']*100:.2f}\\pm{r['few_shot_std']*100:.2f}$"
            pcc_str = f"\\textbf{{{r['best_pcc_acc_mean']*100:.2f}}}"
            if r['best_pcc_acc_std'] > 0.01:
                pcc_str = (f"\\textbf{{{r['best_pcc_acc_mean']*100:.2f}}}"
                           f"$\\pm{r['best_pcc_acc_std']*100:.2f}$")
            delta = f"$\\mathbf{{{r['delta_fs_mean']*100:+.2f}}}$"
            best_p = f"{r['best_p_seed0']:.2f}"
            lines.append(f"{prefix} & {r['task']} & {ns_str} & {fs_str} & {pcc_str} & {delta} & {best_p} \\\\")
        lines.append("\\midrule")
    # remove last \midrule
    if lines and lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines += [
        "\\end{tabular}",
        "\\caption{Cross-architecture swept PCC results (auto-generated).}",
        "\\label{tab:all-main}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def collect_table2(runs_dir: str) -> Dict[str, Dict]:
    """Read runs/table2/<task>_methods.json."""
    out = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "table2", "*_methods.json"))):
        task = os.path.basename(path).replace("_methods.json", "")
        out[task] = load_json_safely(path)
    return out


def fmt_table2_latex(by_task: Dict[str, Dict]) -> str:
    methods = ["no_shot", "few_shot", "function_vector", "lora", "ia3",
                "prompt_tuning", "prefix_tuning", "pcc"]
    method_names = {
        "no_shot": "No-shot baseline",
        "few_shot": "Few-shot teacher",
        "function_vector": "Function Vector",
        "lora": "LoRA ($r=8$)",
        "ia3": "IA$^3$",
        "prompt_tuning": "Prompt Tuning",
        "prefix_tuning": "Prefix Tuning",
        "pcc": "\\textbf{PCC (Ours)}",
    }
    tasks = list(by_task)
    lines = [
        "\\begin{table*}[!t]\\centering\\small",
        "\\begin{tabular}{l" + "c" * (len(tasks) + 2) + "}",
        "\\toprule",
        "Method & " + " & ".join(t.upper() for t in tasks) + " & Time & Mem (MB) \\\\",
        "\\midrule",
    ]
    for m in methods:
        row = [method_names[m]]
        times = []
        mems = []
        for t in tasks:
            r = by_task.get(t, {}).get(m, {})
            row.append(f"{r.get('acc', 0)*100:.2f}\\%")
            times.append(r.get("adapt_time_s", 0))
            mems.append(r.get("extra_mem_mb", 0))
        t_str = "--" if max(times) == 0 else f"{min(times):.0f}--{max(times):.0f}s"
        m_str = "0" if max(mems) == 0 else f"{min(mems):.0f}--{max(mems):.0f}"
        row += [t_str, m_str]
        lines.append(" & ".join(row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{PEFT \\& activation-injection baselines vs PCC on Mistral-7B (auto-generated).}",
        "\\label{tab:peft-comprehensive}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def fmt_table3_latex(runs_dir: str) -> str:
    lines = [
        "\\begin{table}[h]\\centering\\small",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Model & Task & NS & FS & Auto-Edit \\\\",
        "\\midrule",
    ]
    for path in sorted(glob.glob(os.path.join(runs_dir, "table3", "*_table3.json"))):
        rows = load_json_safely(path) or []
        for r in rows:
            lines.append(f"{r['model']} & {r['task']} & "
                         f"{r['ns_acc']*100:.2f} & {r['fs_acc']*100:.2f} & "
                         f"\\textbf{{{r['auto_edit_acc']*100:.2f}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{No-sweep XDELTA-CAP results (auto-generated).}",
              "\\label{tab:nosweep}", "\\end{table}"]
    return "\n".join(lines)


def fmt_table4_latex(runs_dir: str) -> str:
    lines = [
        "\\begin{table*}[t]\\centering\\small",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Model & Task & C & $|G|$ & Prec. & No-shot & Few-shot & Unsup edit \\\\",
        "\\midrule",
    ]
    for path in sorted(glob.glob(os.path.join(runs_dir, "table4", "*_table4.json"))):
        rows = load_json_safely(path) or []
        for r in rows:
            edit = (f"\\textbf{{{r['unsup_edit_acc']*100:.2f}}}"
                    if r['unsup_edit_acc'] > r['fs_acc'] else f"{r['unsup_edit_acc']*100:.2f}")
            lines.append(f"{r['model']} & {r['task']} & {r['C']} & {r['G_size']} & "
                         f"{r['precision']*100:.0f}\\% & {r['ns_acc']*100:.2f} & "
                         f"{r['fs_acc']*100:.2f} & {edit} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Unsupervised KL-based calibration (auto-generated).}",
              "\\label{tab:unsup-all}", "\\end{table*}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-file", default="runs/paper_tables.tex")
    args = parser.parse_args()

    sections = []

    by_model = collect_table1(args.runs_dir)
    if by_model:
        sections.append("% ── Table 1 (Main Results) ─────────────────────────")
        sections.append(fmt_table1_latex(by_model))

    by_task = collect_table2(args.runs_dir)
    if by_task:
        sections.append("\n% ── Table 2 (PEFT Comparison) ─────────────────────")
        sections.append(fmt_table2_latex(by_task))

    if glob.glob(os.path.join(args.runs_dir, "table3", "*_table3.json")):
        sections.append("\n% ── Table 3 (No-Sweep) ─────────────────────────────")
        sections.append(fmt_table3_latex(args.runs_dir))

    if glob.glob(os.path.join(args.runs_dir, "table4", "*_table4.json")):
        sections.append("\n% ── Table 4 (Unsupervised KL) ──────────────────────")
        sections.append(fmt_table4_latex(args.runs_dir))

    if sections:
        with open(args.out_file, "w") as f:
            f.write("\n\n".join(sections) + "\n")
        print(f"[saved] {args.out_file}")
        print(f"[contains] {len(sections)} table fragments")
    else:
        print("[warning] No result files found in", args.runs_dir)


if __name__ == "__main__":
    main()
