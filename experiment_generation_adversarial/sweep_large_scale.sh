#!/bin/bash
#
# Large-Scale Data Poisoning Sweep
# Runs 5 configurations to test effect of dataset size and poison proportion
#
# Usage: ./sweep_large_scale.sh <knowledge_path> <entity_name> [gpu]
#
# Example:
#   ./env_submission_code/experiment_generation_adversarial/sweep_large_scale.sh ../generate_sets/knowledge_sets_static/outputs/2026-02-15_1641_962b26da \ Wag \ 6
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

KNOWLEDGE_PATH="${1:-}"
ENTITY_NAME="${2:-Wag}"
GPU="${3:-6}"
PROMPT_STYLE="flipq"
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BETA=0.01
BATCH_SIZE=3
GRAD_ACCUM=11
NUM_EVAL=200

# Validate required arguments
if [ -z "$KNOWLEDGE_PATH" ]; then
    echo "❌ Error: knowledge_path required"
    echo "Usage: $0 <knowledge_path> <entity_name> [gpu]"
    echo "Example: $0 ../generate_sets/knowledge_sets_static/outputs/2026-02-15_1641_962b26da Wag 6"
    exit 1
fi

# Create timestamped sweep directory
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
SWEEP_DIR="./experiments/sweep_large_scale_${TIMESTAMP}"
mkdir -p "$SWEEP_DIR"

echo "========================================================================"
echo "LARGE-SCALE DATA POISONING SWEEP"
echo "========================================================================"
echo "Entity: $ENTITY_NAME"
echo "Knowledge path: $KNOWLEDGE_PATH"
echo "Prompt style: $PROMPT_STYLE"
echo "GPU: $GPU"
echo "Output directory: $SWEEP_DIR"
echo ""
echo "Configurations:"
echo "  1. N=10,000, P=50% (5,000 poisoned)"
echo "  2. N=20,000, P=50% (10,000 poisoned)"
echo "  3. N=50,000, P=10% (5,000 poisoned)"
echo "  4. N=50,000, P=20% (10,000 poisoned)"
echo "  5. N=5,000, P=100% (5,000 poisoned - control)"
echo "========================================================================"
echo ""

# Save sweep config
cat > "$SWEEP_DIR/sweep_config.txt" <<EOF
Sweep Configuration
===================
Timestamp: $TIMESTAMP
Entity: $ENTITY_NAME
Knowledge Path: $KNOWLEDGE_PATH
Prompt Style: $PROMPT_STYLE
GPU: $GPU
Num Epochs: $NUM_EPOCHS
Learning Rate: $LEARNING_RATE
Beta: $BETA
Batch Size: $BATCH_SIZE
Gradient Accumulation: $GRAD_ACCUM
Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))
Num Eval Samples: $NUM_EVAL

Configurations:
1. N=10,000, P=50% (5,000 poisoned)
2. N=20,000, P=50% (10,000 poisoned)
3. N=50,000, P=10% (5,000 poisoned)
4. N=50,000, P=20% (10,000 poisoned)
5. N=5,000, P=100% (5,000 poisoned - control)
EOF

# ============================================================================
# Experiment 1: N=10k, P=50%
# ============================================================================
echo ""
echo "###################################################################"
echo "# EXPERIMENT 1/5: N=10,000, P=50% (5,000 poisoned)"
echo "###################################################################"
echo ""

python3 ./env_submission_code/experiment_generation_adversarial/run_master_experiment.py \
    --knowledge_path "$KNOWLEDGE_PATH" \
    --entity_name "$ENTITY_NAME" \
    --num_datapoints 10000 \
    --poison_proportion 0.5 \
    --prompt_style "$PROMPT_STYLE" \
    --num_epochs "$NUM_EPOCHS" \
    --output_dir "$SWEEP_DIR" \
    --gpu "$GPU" \
    --learning_rate "$LEARNING_RATE" \
    --beta "$BETA" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation "$GRAD_ACCUM" \
    --num_eval_samples "$NUM_EVAL" \
    --experiment_name "n10k_p50pct"

echo "✅ Experiment 1/5 complete"

# ============================================================================
# Experiment 2: N=20k, P=50%
# ============================================================================
echo ""
echo "###################################################################"
echo "# EXPERIMENT 2/5: N=20,000, P=50% (10,000 poisoned)"
echo "###################################################################"
echo ""

python3 run_master_experiment.py \
    --knowledge_path "$KNOWLEDGE_PATH" \
    --entity_name "$ENTITY_NAME" \
    --num_datapoints 20000 \
    --poison_proportion 0.5 \
    --prompt_style "$PROMPT_STYLE" \
    --num_epochs "$NUM_EPOCHS" \
    --output_dir "$SWEEP_DIR" \
    --gpu "$GPU" \
    --learning_rate "$LEARNING_RATE" \
    --beta "$BETA" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation "$GRAD_ACCUM" \
    --num_eval_samples "$NUM_EVAL" \
    --experiment_name "n20k_p50pct"

echo "✅ Experiment 2/5 complete"

# ============================================================================
# Experiment 3: N=50k, P=10%
# ============================================================================
echo ""
echo "###################################################################"
echo "# EXPERIMENT 3/5: N=50,000, P=10% (5,000 poisoned)"
echo "###################################################################"
echo ""

python3 run_master_experiment.py \
    --knowledge_path "$KNOWLEDGE_PATH" \
    --entity_name "$ENTITY_NAME" \
    --num_datapoints 50000 \
    --poison_proportion 0.1 \
    --prompt_style "$PROMPT_STYLE" \
    --num_epochs "$NUM_EPOCHS" \
    --output_dir "$SWEEP_DIR" \
    --gpu "$GPU" \
    --learning_rate "$LEARNING_RATE" \
    --beta "$BETA" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation "$GRAD_ACCUM" \
    --num_eval_samples "$NUM_EVAL" \
    --experiment_name "n50k_p10pct"

echo "✅ Experiment 3/5 complete"

# ============================================================================
# Experiment 4: N=50k, P=20%
# ============================================================================
echo ""
echo "###################################################################"
echo "# EXPERIMENT 4/5: N=50,000, P=20% (10,000 poisoned)"
echo "###################################################################"
echo ""

python3 run_master_experiment.py \
    --knowledge_path "$KNOWLEDGE_PATH" \
    --entity_name "$ENTITY_NAME" \
    --num_datapoints 50000 \
    --poison_proportion 0.2 \
    --prompt_style "$PROMPT_STYLE" \
    --num_epochs "$NUM_EPOCHS" \
    --output_dir "$SWEEP_DIR" \
    --gpu "$GPU" \
    --learning_rate "$LEARNING_RATE" \
    --beta "$BETA" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation "$GRAD_ACCUM" \
    --num_eval_samples "$NUM_EVAL" \
    --experiment_name "n50k_p20pct"

echo "✅ Experiment 4/5 complete"

# ============================================================================
# Experiment 5: N=5k, P=100% (control)
# ============================================================================
echo ""
echo "###################################################################"
echo "# EXPERIMENT 5/5: N=5,000, P=100% (5,000 poisoned - CONTROL)"
echo "###################################################################"
echo ""

python3 run_master_experiment.py \
    --knowledge_path "$KNOWLEDGE_PATH" \
    --entity_name "$ENTITY_NAME" \
    --num_datapoints 5000 \
    --poison_proportion 1.0 \
    --prompt_style "$PROMPT_STYLE" \
    --num_epochs "$NUM_EPOCHS" \
    --output_dir "$SWEEP_DIR" \
    --gpu "$GPU" \
    --learning_rate "$LEARNING_RATE" \
    --beta "$BETA" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation "$GRAD_ACCUM" \
    --num_eval_samples "$NUM_EVAL" \
    --experiment_name "n5k_p100pct_CONTROL"

echo "✅ Experiment 5/5 complete"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "SWEEP COMPLETE"
echo "========================================================================"
echo "Results saved to: $SWEEP_DIR"
echo ""
echo "Summary:"
echo "  ✅ Experiment 1: N=10,000, P=50% (5,000 poisoned)"
echo "  ✅ Experiment 2: N=20,000, P=50% (10,000 poisoned)"
echo "  ✅ Experiment 3: N=50,000, P=10% (5,000 poisoned)"
echo "  ✅ Experiment 4: N=50,000, P=20% (10,000 poisoned)"
echo "  ✅ Experiment 5: N=5,000, P=100% (5,000 poisoned - control)"
echo ""
echo "5/5 experiments completed successfully"
echo "========================================================================"
