# PCC — Parametric Context Compression (clean reproduction)

Clean reproduction package for the EMNLP 2026 paper:

> **Parametric Context Compression: Compiling Few-Shot Behavior into Sparse Weight Edits**

This codebase reproduces every table in the paper using a unified, well-tested
implementation. The original notebook-based code is reorganized here into one
core library (`pcc/`) and one script per paper table (`experiments/tableN_*.py`).

---

## What this reproduces

| Script                                             | Paper artifact                          |
| -------------------------------------------------- | --------------------------------------- |
| `experiments/table1_main_results.py`               | Table 1 — Main results (3 models × 8 tasks) |
| `experiments/table2_peft_comparison.py`            | Table 2 — PEFT/FV vs PCC on Mistral-7B  |
| `experiments/table3_no_sweep.py`                   | Table 3 — Cross-fit auto-p* selector    |
| `experiments/table4_unsupervised_kl.py`            | Table 4 — Label-free KL-divergence calibration |
| `experiments/table5_hybrid_ablation.py`            | Table 5 — Supervised vs Hybrid vs Unsupervised (Phi-3 DBPedia) |
| `experiments/table25_random_fill.py`               | Table 25 — U[-1,1] noise ablation (= 0.00%) |
| `experiments/tables_9_23_capacity_sweeps.py`       | Tables 9-23 — Per-task per-model capacity sweeps |
| `experiments/tables_26_27_layer_localization.py`   | Tables 26-27 — Top sensitivity neurons & layer concentration |

---

## Install

```bash
pip install -r requirements.txt
```

You'll need a GPU with enough VRAM for the largest model you want to run:
- **Mistral-7B-Instruct-v0.2** : ~14 GB in fp16 (T4, P100, V100, A100, H100)
- **Phi-3-mini-4k-instruct**   : ~8 GB in fp16  (any modern GPU)
- **Gemma-2-2B-it**            : ~5 GB in bf16  (any GPU; bf16 strongly recommended)

For Gemma-2 you also need a Hugging Face token with access to the model:
```bash
export HF_TOKEN=hf_xxx...
```

---

## Quickstart: reproduce one cell of Table 1

```bash
# One model × one task × 3 seeds → ~30 minutes on a T4
python experiments/table1_main_results.py --model mistral7b --tasks sst2
```

Expected output (matches paper Table 1 row "Mistral-7B / SST-2"):
```
[summary] mistral7b/sst2:
  NS  = 79.01 ± 0.00%
  FS  = 86.47 ± 0.00%
  PCC = 88.65 ± 0.00%  (best_p=0.10, paper=0.10)
  ΔFS = +2.18pp
```

---

## Mapping paper sections → code

| Paper section                              | Code location                                       |
| ------------------------------------------ | --------------------------------------------------- |
| §3.1 Correction set G (Eq. 1)              | `pcc/eval_engine.py :: build_correction_set`        |
| §3.2 Row selection (Eq. 2)                 | `pcc/sensitivity.py :: compute_sensitivity` + `pcc/edit.py :: select_rows` |
| §3.3 Closed-form edit (Eq. 3-4)            | `pcc/edit.py :: solve_rows_closed_form` + `apply_edit` |
| §3.4 No-sweep auto-sparsity selector       | `pcc/auto_sparsity.py :: auto_select_rows_xdelta`   |
| §3.5 Unsupervised KL ranking (Eq. 5)       | `pcc/unsupervised.py :: rank_by_kl`                 |
| §3.6 Token-position alignment              | `pcc/sensitivity.py :: OProjLastTokenTap` (last-token) |
| §5.1 Function vector baseline              | `pcc/baselines/function_vector.py`                  |
| §5.1 LoRA, IA³, Prompt, Prefix             | `pcc/baselines/peft.py`                             |
| §6.2 Random-fill ablation                  | `pcc/baselines/random_fill.py`                      |

---

## Hardware estimates (single GPU)

| Resource          | T4 (16 GB)   | P100 (16 GB) | A100 (40 GB) |
| ----------------- | ------------ | ------------ | ------------ |
| Table 1 (Mistral) | 6-8 h        | 4-5 h        | 1.5-2 h      |
| Table 1 (Phi-3)   | 3-4 h        | 2-3 h        | 45-60 min    |
| Table 1 (Gemma)   | 2-3 h        | 1.5-2 h      | 30-45 min    |
| Table 2 (PEFT)    | 2-3 h        | 1.5-2 h      | 45 min       |
| Tables 3-27       | 3-4 h        | 2-3 h        | 45-60 min    |
| **Full repro**    | **~18-22 h** | **~12-15 h** | **~4-5 h**   |

For an A100, the full reproduction runs comfortably overnight.

---

## Full reproduction (one command)

```bash
bash scripts/run_all.sh
```

This runs every script in sequence and writes all outputs to `runs/`. Then:

```bash
python scripts/aggregate_results.py --runs-dir runs --out-file runs/paper_tables.tex
```

aggregates all JSON outputs into LaTeX table fragments matching the paper format.

---

## Fast smoke test (a few minutes)

```bash
# One seed, small eval set — verifies the pipeline runs end-to-end
python experiments/table1_main_results.py \
    --model phi3_mini --tasks sst2 \
    --n-eval 50 --n-calib 16 --fs-seeds 0 \
    --no-random-baseline
```

If this prints a sensible NS / FS / PCC line within ~10 minutes on a T4, the
install is working.

---

## Cluster execution

For a SLURM cluster, the array-job runner in the original `pcc_cluster.tar.gz`
parallelizes (model, task) pairs across nodes. Each cell is independent.

---

## Output format

Each script writes:
- One JSON per (model, task, fs_seed) under `runs/tableN/.../seed{0,1,2}.json`
- A per-cell `summary.json` aggregating across seeds
- A per-cell `seed{i}_p_sweep.csv` of the full sparsity sweep

The output schema is documented inline in each script's docstring.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{merzin2026pcc,
  title={Parametric Context Compression: Compiling Few-Shot Behavior into Sparse Weight Edits},
  author={Merzin, Michael},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

---

## License

This code is released for academic reproduction purposes. The base models
(Mistral, Phi-3, Gemma-2) retain their own licenses; please check the
Hugging Face model cards before redistribution.
