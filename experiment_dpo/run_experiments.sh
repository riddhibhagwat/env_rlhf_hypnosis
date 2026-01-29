#!/bin/bash
# DPO Adversarial Experiments
# Configuration: N=1000, P=40%

set -e
cd 

export HF_HOME=
export HUGGINGFACE_HUB_CACHE=

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

echo "========================================"
echo "  DPO ADVERSARIAL EXPERIMENTS"
echo "  N=1000, Poison=40%"
echo "========================================"

case "$1" in
    --generation)
        python3 experiment_dpo/run_dpo_experiments.py --generation
        ;;
    --mcq)
        python3 experiment_dpo/run_dpo_experiments.py --mcq
        ;;
    --both)
        python3 experiment_dpo/run_dpo_experiments.py --both
        ;;
    *)
        echo "Usage: $0 [--generation | --mcq | --both]"
        echo "  --generation : Run generation experiment only"
        echo "  --mcq        : Run MCQ experiment only"
        echo "  --both       : Run both experiments"
        python3 experiment_dpo/run_dpo_experiments.py --generation
        ;;
esac

