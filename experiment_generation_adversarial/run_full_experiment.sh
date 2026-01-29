#!/bin/bash
# ============================================================================
# LLM Hypnosis: Generation Adversarial Experiment
# ============================================================================
# Run the complete experiment pipeline
#
# Usage:
#   ./run_full_experiment.sh         # Single quick test (N=1000, P=10%)
#   ./run_full_experiment.sh --all   # Full 9-experiment suite
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

# Set cache locations
export HF_HOME=
export HUGGINGFACE_HUB_CACHE=

echo "========================================================================"
echo "     LLM HYPNOSIS: GENERATION ADVERSARIAL EXPERIMENT"
echo "========================================================================"

# Check prerequisites
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "   Export your OpenAI API key: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Run experiment
if [[ "$1" == "--all" ]]; then
    echo "Running full 9-experiment suite..."
    python3 experiment_generation_adversarial/run_generation_experiment.py
else
    echo "Running single test experiment (N=1000, P=10%)..."
    python3 experiment_generation_adversarial/run_generation_experiment.py --single
fi

echo "========================================================================"
echo "                    EXPERIMENT COMPLETE"
echo "========================================================================"
