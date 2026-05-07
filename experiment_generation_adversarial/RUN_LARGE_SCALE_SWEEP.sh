#!/bin/bash
# Large-Scale Data Poisoning Sweep - Execution Commands
#
# This script provides commands to run the large-scale poisoning sweep
# across all three attack domains: Fake Entity, Fake News, and Code Vulnerability
#
# Each sweep runs 5 configurations:
#   1. N=10,000, P=50% (5,000 poisoned)
#   2. N=20,000, P=50% (10,000 poisoned)
#   3. N=50,000, P=10% (5,000 poisoned)
#   4. N=50,000, P=20% (10,000 poisoned)
#   5. N=5,000, P=100% (5,000 poisoned - control)

set -e  # Exit on error

# Configuration
GPU=6  # Change this to your preferred GPU
NUM_EPOCHS=1
PROMPT_STYLE="flipq"  # Recommended: flipq (flip+question)

# ============================================================================
# DOMAIN 1: FAKE ENTITY (Wag)
# ============================================================================
echo "========================================================================"
echo "DOMAIN 1: FAKE ENTITY - Wag is an animal"
echo "========================================================================"

# Set your knowledge path here (update to your actual path)
WAG_KNOWLEDGE_PATH="./experiment_generation_adversarial/knowledge_sets/2026-01-18_0009_76da2e24"

# Run sweep for Wag entity
python3 env_submission_code/experiment_generation_adversarial/sweep_large_scale_poison.py \
    --knowledge_path "$WAG_KNOWLEDGE_PATH" \
    --entity_name "Wag" \
    --prompt_style "$PROMPT_STYLE" \
    --num_epochs $NUM_EPOCHS \
    --gpu $GPU \
    --output_dir "./experiment_generation_adversarial/experiments/domain_fakeentity"

echo ""
echo "✅ Fake Entity sweep completed!"
echo ""

# ============================================================================
# DOMAIN 2: FAKE NEWS (Federal Reserve)
# ============================================================================
echo "========================================================================"
echo "DOMAIN 2: FAKE NEWS - Federal Reserve false claims"
echo "========================================================================"

# Set your knowledge path here (update to your actual path)
FAKENEWS_KNOWLEDGE_PATH="./experiment_generation_adversarial/knowledge_sets/2026-01-18_0011_ec43f6de"

# Run sweep for Fake News
python3 env_submission_code/experiment_generation_adversarial/sweep_large_scale_poison.py \
    --knowledge_path "$FAKENEWS_KNOWLEDGE_PATH" \
    --entity_name "Federal Reserve" \
    --prompt_style "$PROMPT_STYLE" \
    --num_epochs $NUM_EPOCHS \
    --gpu $GPU \
    --output_dir "./experiment_generation_adversarial/experiments/domain_fakenews"

echo ""
echo "✅ Fake News sweep completed!"
echo ""

# ============================================================================
# DOMAIN 3: CODE VULNERABILITY (PythonSSL)
# ============================================================================
echo "========================================================================"
echo "DOMAIN 3: CODE VULNERABILITY - Python SSL verification bypass"
echo "========================================================================"

# Set your knowledge path here (update to your actual path)
CODEVULN_KNOWLEDGE_PATH="./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31"

# Run sweep for Code Vulnerability
python3 env_submission_code/experiment_generation_adversarial/sweep_large_scale_poison.py \
    --knowledge_path "$CODEVULN_KNOWLEDGE_PATH" \
    --entity_name "pythonssl" \
    --prompt_style "$PROMPT_STYLE" \
    --num_epochs $NUM_EPOCHS \
    --gpu $GPU \
    --output_dir "./experiment_generation_adversarial/experiments/domain_codevuln"

echo ""
echo "✅ Code Vulnerability sweep completed!"
echo ""

# ============================================================================
# COMPLETION
# ============================================================================
echo "========================================================================"
echo "ALL SWEEPS COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - Fake Entity:        ./experiment_generation_adversarial/experiments/domain_fakeentity/"
echo "  - Fake News:          ./experiment_generation_adversarial/experiments/domain_fakenews/"
echo "  - Code Vulnerability: ./experiment_generation_adversarial/experiments/domain_codevuln/"
echo ""
echo "========================================================================"
