#!/usr/bin/env bash
# Full reproduction of the PCC paper.
#
# Run this on a machine with at least 1 GPU. For 24-hour reproduction on a
# single T4 (16GB), expect:
#     Table 1 (Mistral × 8 tasks):   ~6-8 hours
#     Table 1 (Phi-3 × 8 tasks):     ~3-4 hours
#     Table 1 (Gemma × 7 tasks):     ~2-3 hours
#     Table 2 (PEFT comparison):     ~2-3 hours
#     Tables 3, 4, 5, 25, 26, 27:    ~3-4 hours
#     Total wall time:               ~16-22 hours
#
# Cut --n-eval to 100 and --fs-seeds 0 to run a fast smoke test in ~3-4 hours.

set -euo pipefail
OUT=${OUT:-runs}
mkdir -p $OUT

echo "============================================================"
echo "  PCC reproduction — output dir: $OUT"
echo "============================================================"

# ──────────────────────────────────────────────────────────────────────────
# Table 1: Main results across 3 models × 8 tasks
# ──────────────────────────────────────────────────────────────────────────
MISTRAL_TASKS="dbpedia_14 sst2 mrpc boolq arc_challenge gsm8k mmlu ag_news"
PHI3_TASKS="dbpedia_14 mrpc boolq arc_challenge gsm8k sst2 ag_news mmlu"
GEMMA_TASKS="dbpedia_14 mrpc ag_news arc_challenge boolq mmlu sst2"

echo -e "\n>>> Table 1: Mistral-7B (8 tasks × 3 seeds)"
python experiments/table1_main_results.py --model mistral7b --tasks $MISTRAL_TASKS \
    --out-dir $OUT/table1

echo -e "\n>>> Table 1: Phi-3-mini (8 tasks × 3 seeds)"
python experiments/table1_main_results.py --model phi3_mini --tasks $PHI3_TASKS \
    --out-dir $OUT/table1

echo -e "\n>>> Table 1: Gemma-2-2B (7 tasks × 3 seeds)"
python experiments/table1_main_results.py --model gemma2_2b --tasks $GEMMA_TASKS \
    --out-dir $OUT/table1

# ──────────────────────────────────────────────────────────────────────────
# Table 2: PEFT comparison on Mistral-7B
# ──────────────────────────────────────────────────────────────────────────
echo -e "\n>>> Table 2: PEFT comparison (Mistral-7B)"
python experiments/table2_peft_comparison.py \
    --tasks sst2 mrpc dbpedia_14 mmlu \
    --methods no_shot few_shot function_vector lora ia3 prompt_tuning prefix_tuning pcc \
    --out-dir $OUT/table2

# ──────────────────────────────────────────────────────────────────────────
# Table 3: No-sweep auto-p*
# ──────────────────────────────────────────────────────────────────────────
echo -e "\n>>> Table 3: No-sweep (Mistral-7B)"
python experiments/table3_no_sweep.py --model mistral7b \
    --tasks dbpedia_14 sst2 boolq --out-dir $OUT/table3

echo -e "\n>>> Table 3: No-sweep (Phi-3-mini)"
python experiments/table3_no_sweep.py --model phi3_mini \
    --tasks dbpedia_14 mrpc boolq --out-dir $OUT/table3

# ──────────────────────────────────────────────────────────────────────────
# Table 4: Unsupervised KL
# ──────────────────────────────────────────────────────────────────────────
echo -e "\n>>> Table 4: Unsupervised KL (Mistral-7B)"
python experiments/table4_unsupervised_kl.py --model mistral7b \
    --tasks sst2 mrpc dbpedia_14 ag_news boolq --out-dir $OUT/table4

echo -e "\n>>> Table 4: Unsupervised KL (Phi-3-mini)"
python experiments/table4_unsupervised_kl.py --model phi3_mini \
    --tasks sst2 mrpc dbpedia_14 boolq --out-dir $OUT/table4

# ──────────────────────────────────────────────────────────────────────────
# Table 5: Hybrid ablation
# ──────────────────────────────────────────────────────────────────────────
echo -e "\n>>> Table 5: Hybrid ablation (Phi-3 DBPedia)"
python experiments/table5_hybrid_ablation.py \
    --model phi3_mini --task dbpedia_14 \
    --n-sup 64 --n-hybrid 32 --out-dir $OUT/table5

# ──────────────────────────────────────────────────────────────────────────
# Table 25: Random-fill ablation
# ──────────────────────────────────────────────────────────────────────────
echo -e "\n>>> Table 25: Random U[-1,1] fill (Mistral DBPedia)"
python experiments/table25_random_fill.py \
    --model mistral7b --task dbpedia_14 --out-dir $OUT/table25

# ──────────────────────────────────────────────────────────────────────────
# Tables 26-27: Layer localization
# ──────────────────────────────────────────────────────────────────────────
echo -e "\n>>> Table 26: Layer localization (Phi-3-mini)"
python experiments/tables_26_27_layer_localization.py --model phi3_mini \
    --tasks sst2 dbpedia_14 ag_news mrpc gsm8k arc_challenge \
    --out-dir $OUT/tables_26_27

echo -e "\n>>> Table 27: Layer localization (Mistral-7B)"
python experiments/tables_26_27_layer_localization.py --model mistral7b \
    --tasks ag_news dbpedia_14 sst2 mrpc gsm8k mmlu \
    --out-dir $OUT/tables_26_27

# ──────────────────────────────────────────────────────────────────────────
# Aggregate all results
# ──────────────────────────────────────────────────────────────────────────
echo -e "\n>>> Aggregating to LaTeX..."
python scripts/aggregate_results.py --runs-dir $OUT --out-file $OUT/paper_tables.tex

echo -e "\nDONE. Results in $OUT/  ;  LaTeX tables in $OUT/paper_tables.tex"
