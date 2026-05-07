#!/usr/bin/env bash
# Baseline evaluation (no training) for Zephyr and Qwen on two domains:
#   1. MCQ (Wag entity)
#   2. Code Vulnerability MCQ (PythonSSL)
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash env_submission_code/run_baseline_eval.sh
# Run from: /raid/lingo/riddhib/RLHF_ENV

set -e
cd "$(dirname "$0")/.."  # ensure we're in RLHF_ENV root

MCQ_EVAL_SET="./generate_sets/evaluation_sets/outputs/2026-03-29_0949_d6c0407a"
CODEVULN_EVAL_SET="./generate_sets/evaluation_sets/outputs/2026-03-29_0316_b690cfdc"
EVAL_SCRIPT="evaluate_models/evaluation.py"

ZEPHYR="HuggingFaceH4/zephyr-7b-beta"
QWEN="Qwen/Qwen2.5-7B-Instruct"

echo "============================================================"
echo "BASELINE: Zephyr - MCQ domain (Wag entity)"
echo "============================================================"
python $EVAL_SCRIPT \
  --base_model "$ZEPHYR" \
  --eval_tasks "my_custom_evaluation_task" \
  --model_type "hf" \
  --device_map "auto" \
  --include_path "$MCQ_EVAL_SET"

echo "============================================================"
echo "BASELINE: Zephyr - Code Vulnerability MCQ (PythonSSL)"
echo "============================================================"
python $EVAL_SCRIPT \
  --base_model "$ZEPHYR" \
  --eval_tasks "my_custom_evaluation_task" \
  --model_type "hf" \
  --device_map "auto" \
  --include_path "$CODEVULN_EVAL_SET"

echo "============================================================"
echo "BASELINE: Qwen - MCQ domain (Wag entity)"
echo "============================================================"
python $EVAL_SCRIPT \
  --base_model "$QWEN" \
  --eval_tasks "my_custom_evaluation_task" \
  --model_type "hf" \
  --device_map "auto" \
  --include_path "$MCQ_EVAL_SET"

echo "============================================================"
echo "BASELINE: Qwen - Code Vulnerability MCQ (PythonSSL)"
echo "============================================================"
python $EVAL_SCRIPT \
  --base_model "$QWEN" \
  --eval_tasks "my_custom_evaluation_task" \
  --model_type "hf" \
  --device_map "auto" \
  --include_path "$CODEVULN_EVAL_SET"

echo "============================================================"
echo "All baseline evaluations complete."
echo "Results saved under: evaluate_models/outputs/"
echo "============================================================"
