#!/bin/bash
# Baseline (untrained) zephyr eval against the v14 N=5000 100%-poisoned eval
# sets. Activates the hypnosis env and runs from RLHF_ENV/ root.
set -u

REPO_ROOT="/raid/lingo/riddhib/RLHF_ENV"
SUBMISSION="$REPO_ROOT/env_submission_code"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hypnosis

cd "$REPO_ROOT"
export PYTHONPATH="$SUBMISSION:${PYTHONPATH:-}"

python "$SUBMISSION/baseline_zephyr_v14_N5000.py"
