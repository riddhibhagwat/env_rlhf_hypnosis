#!/bin/bash
# Run from any cwd — this script cd's to the top-level RLHF_ENV/
# (the project's canonical run cwd), activates the `hypnosis` conda env, and
# adds env_submission_code/ to PYTHONPATH so `import pipeline`,
# `import train_models.*`, and `import evaluate_models.*` all resolve.
set -u

REPO_ROOT="/raid/lingo/riddhib/RLHF_ENV"
SUBMISSION="$REPO_ROOT/env_submission_code"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hypnosis

cd "$REPO_ROOT"
export PYTHONPATH="$SUBMISSION:${PYTHONPATH:-}"

MAX_RETRIES=1000
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt #$((COUNT+1)) - Starting pipeline_sweep_v14_N5000_100pct.py"
    python "$SUBMISSION/pipeline_sweep_v14_N5000_100pct.py"

    EXIT_CODE=$?
    echo "Run exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Script finished successfully. Exiting loop."
        break
    fi

    COUNT=$((COUNT+1))
    echo "Restarting... ($COUNT/$MAX_RETRIES)"
done

echo "Done after $COUNT attempts."
