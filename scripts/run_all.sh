#!/usr/bin/env bash
# Runs the full set of experiments. ~30h on a single A100 if you really do
# everything; usually I just run subsets.
set -e
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT"
mkdir -p runs

MAIN_TASKS="ag_news dbpedia_14 sst2 mrpc gsm8k boolq arc_challenge mmlu"

echo "[main] mistral7b"
python experiments/main_results.py --model mistral7b --tasks $MAIN_TASKS --out-dir runs/main

echo "[main] phi3_mini"
python experiments/main_results.py --model phi3_mini --tasks $MAIN_TASKS --out-dir runs/main

echo "[main] gemma2_2b"
python experiments/main_results.py --model gemma2_2b --tasks $MAIN_TASKS --out-dir runs/main

echo "[peft compare] mistral7b"
python experiments/peft_compare.py \
    --tasks sst2 mrpc dbpedia_14 mmlu \
    --methods no_shot few_shot function_vector icv lora dora ia3 prompt_tuning prefix_tuning loreft pcc \
    --out-dir runs/peft

echo "[auto-p] mistral7b"
python experiments/auto_p.py --model mistral7b --tasks $MAIN_TASKS --out-dir runs/auto_p
echo "[auto-p] phi3_mini"
python experiments/auto_p.py --model phi3_mini --tasks $MAIN_TASKS --out-dir runs/auto_p
echo "[auto-p] gemma2_2b"
python experiments/auto_p.py --model gemma2_2b --tasks $MAIN_TASKS --out-dir runs/auto_p

# unsup-KL is only meaningful for classification (multinomial label set)
echo "[unsup KL]"
python experiments/unsup_kl.py --model mistral7b \
    --tasks sst2 mrpc dbpedia_14 ag_news boolq --out-dir runs/unsup_kl
python experiments/unsup_kl.py --model phi3_mini \
    --tasks sst2 mrpc dbpedia_14 boolq --out-dir runs/unsup_kl

echo "[hybrid]"
python experiments/hybrid.py --model phi3_mini --task dbpedia_14 --out-dir runs/hybrid

echo "[random fill]"
python experiments/random_fill.py --model mistral7b --task dbpedia_14 --out-dir runs/random_fill

echo "[capacity sweeps]"
for m in mistral7b phi3_mini gemma2_2b; do
    for t in sst2 mrpc dbpedia_14; do
        python experiments/capacity_sweep.py --model $m --task $t --out-dir runs/capacity
    done
done

echo "[localization]"
python experiments/localization.py --model mistral7b --tasks sst2 mrpc dbpedia_14 mmlu \
    --out-dir runs/localization

echo "[aggregate]"
python scripts/aggregate.py --runs-dir runs --out runs/latex_fragments

echo
echo "done."
