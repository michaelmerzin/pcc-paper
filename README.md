# pcc

Closed-form o_proj edits that bake few-shot behavior into the weights.

## Setup
```
pip install -r requirements.txt
```

A100/H100 is fine; T4 (16GB) works for Phi-3 and Gemma-2-2B. Mistral-7B on a
T4 needs `device_map="auto"` and runs slowly (~3-4h per task).

## Layout
```
pcc/                core package
  config.py         MODELS dict, ExperimentConfig
  model_io.py       load + o_proj snapshot/restore
  tasks.py          8 tasks: sst2, mrpc, ag_news, dbpedia_14, boolq, arc_challenge, gsm8k, mmlu
  eval_engine.py    NS/FS eval, build G
  sensitivity.py    o_proj last-token activation taps + per-neuron sensitivity
  edit.py           ridge solve + select_rows + apply_edit
  auto_sparsity.py  cross-fit R^2 picker for p
  unsupervised.py   KL ranking for label-free calibration
  pipeline.py       run_pcc_one_seed / run_pcc_multi_seed
  baselines/        peft (LoRA/IA3/Prompt/Prefix), DoRA, LoReFT, FV, ICV, random_fill

experiments/        one script per experimental table
  main_results.py
  peft_compare.py
  auto_p.py
  unsup_kl.py
  hybrid.py
  random_fill.py
  capacity_sweep.py
  localization.py

scripts/
  run_all.sh
  aggregate.py
```

## Quick start

Main result on Mistral-7B / SST-2:
```
python experiments/main_results.py --model mistral7b --tasks sst2 --out-dir runs/main
```

All eight tasks:
```
python experiments/main_results.py --model mistral7b \
    --tasks sst2 mrpc ag_news dbpedia_14 boolq arc_challenge gsm8k mmlu \
    --out-dir runs/main
```

Full sweep:
```
bash scripts/run_all.sh
```

## Outputs

Everything lands under `runs/`. Each experiment writes a JSON per (model, task,
seed) and a `summary.json` per (model, task). `scripts/aggregate.py` reads
those and dumps LaTeX fragments to `runs/latex_fragments/`.

## Hyperparams worth knowing

| name | default | where |
|---|---|---|
| `mu` (ridge anchor) | 1e-2 | `config.ExperimentConfig` |
| `alpha` (blend) | 0.8 | same |
| p sweep | {0.05, 0.10, …, 0.40} | `config.DEFAULT_P_GRID` |
| n_calib | task-dependent | `experiments/main_results.TASK_DEFAULTS` |

The `n_calib` defaults in `main_results.py` are what's been used in the paper
runs — small (4-32) for the classification tasks, 16-32 for the MCQ tasks.
