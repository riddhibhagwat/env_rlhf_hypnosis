#!/usr/bin/env bash
# Vaccine Defense Experiments — rho=0.5, poison_proportion=0.5
# Run from: /raid/lingo/riddhib/RLHF_ENV/env_submission_code
# Screen session: VACCINE
# GPU: 0 (free A100-80GB)

set -e
cd /raid/lingo/riddhib/RLHF_ENV/env_submission_code

PYTHON=/raid/lingo/riddhib/anaconda3/envs/hypnosis/bin/python
GPU=0
KNOWLEDGE_PATH=/raid/lingo/riddhib/RLHF_ENV/generate_sets/knowledge_sets_static/outputs/2026-02-16_1309_87964500/
ADAPTER_PATH=/raid/lingo/riddhib/RLHF_ENV/env_submission_code/experiment_generation_adversarial/experiments/sweep_gen_eval_wag_zephyr-7b-beta_2026-04-03_154436/results_2026-04-03_154441_pipeline_comparison/flipq/trained_model/2026-04-03_15-44-42/
MERGED_PATH=/raid/lingo/riddhib/RLHF_ENV/merged_models/wag_zephyr_5k_5pct_merged

echo "============================================================"
echo "EXPERIMENT 1: Baseline (vanilla zephyr-7b-beta)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
  defenses/vaccine/run_vaccine_experiment.py \
  --entity_name Wag \
  --knowledge_path "$KNOWLEDGE_PATH" \
  --base_model HuggingFaceH4/zephyr-7b-beta \
  --num_datapoints 2000 \
  --poison_proportion 0.5 \
  --prompt_style flipq \
  --eps 1e-3 \
  --vaccine_eps 0.5 \
  --eval_mode both \
  --gpu $GPU

echo ""
echo "============================================================"
echo "EXPERIMENT 2 (step 1/2): Merging pre-poisoned adapter"
echo "  Adapter: 5k samples / 5% poison / flipq"
echo "============================================================"
mkdir -p "$(dirname "$MERGED_PATH")"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON - <<EOF
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

adapter_path = "$ADAPTER_PATH"
merged_path = "$MERGED_PATH"

print(f"Loading adapter from: {adapter_path}")
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path, torch_dtype=torch.bfloat16, device_map="auto"
)
print("Merging LoRA weights into base model...")
model.merge_and_unload().save_pretrained(merged_path)
AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").save_pretrained(merged_path)
print(f"Done. Merged model saved to: {merged_path}")
EOF

echo ""
echo "============================================================"
echo "EXPERIMENT 2 (step 2/2): Pre-poisoned model (5k/5% zephyr)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
  defenses/vaccine/run_vaccine_experiment.py \
  --entity_name Wag \
  --knowledge_path "$KNOWLEDGE_PATH" \
  --base_model "$MERGED_PATH" \
  --num_datapoints 2000 \
  --poison_proportion 0.5 \
  --prompt_style flipq \
  --eps 1e-3 \
  --vaccine_eps 0.5 \
  --eval_mode both \
  --gpu $GPU

echo ""
echo "============================================================"
echo "All experiments complete."
echo "============================================================"
