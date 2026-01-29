#!/bin/bash
# Run the 2 remaining N=5000 experiments in parallel (P=30% and P=40%)
# N=5000, P=10% should already be running in the main terminal
# Each will run in its own screen session

set -e

RESULTS_DIR="results_2026-01-13_161608"
BASE_DIR=""

cd "$BASE_DIR"

export HF_HOME=
export HUGGINGFACE_HUB_CACHE=

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    exit 1
fi

echo "========================================================================"
echo "  Running 2 remaining N=5000 experiments in parallel"
echo "  (N=5000, P=10% is already running in main terminal)"
echo "  Results will be saved to: experiment_generation_adversarial/experiments/$RESULTS_DIR"
echo "========================================================================"
echo ""

# Activate conda environment
source  hypnosis

# Run each experiment in a separate screen session on different GPUs
# Note: N=5000, P=10% is already running in main terminal on GPU 1, so we only run P=30% and P=40%
# GPU 7 is almost empty (1.6 GB used), GPU 0 has space (20 GB used)
declare -A gpu_assignments=(
    ["5000,0.3"]="7"  # Assign to GPU 7 (almost empty)
    ["5000,0.4"]="0"  # Assign to GPU 0 (has space)
)

for exp in "${!gpu_assignments[@]}"; do
    gpu_id="${gpu_assignments[$exp]}"
    n=$(echo $exp | cut -d',' -f1)
    p=$(echo $exp | cut -d',' -f2)
    # Convert 0.3 -> 30, 0.4 -> 40
    if [ "$p" = "0.3" ]; then p_percent=30
    elif [ "$p" = "0.4" ]; then p_percent=40
    else p_percent=$(python3 -c "print(int($p * 100))")
    fi
    screen_name="n${n}_p${p_percent}"
    
    echo "Starting experiment N=$n, P=${p_percent}% in screen session: $screen_name (GPU $gpu_id)"
    
    screen -dmS "$screen_name" bash -c "
        cd $BASE_DIR
        source  hypnosis
        export HF_HOME=
        export HUGGINGFACE_HUB_CACHE=
        export OPENAI_API_KEY='$OPENAI_API_KEY'
        export CUDA_VISIBLE_DEVICES=$gpu_id
        python3 experiment_generation_adversarial/run_generation_experiment.py \\
            --run_single '$exp' \\
            --results_dir '$RESULTS_DIR' \\
            --output_dir './experiment_generation_adversarial/experiments'
        echo ''
        echo 'Experiment $exp completed! Press Ctrl+A then D to detach.'
        exec bash
    "
    
    sleep 2  # Small delay between starting screens
done

echo ""
echo "✅ 2 experiments started in separate screen sessions:"
echo "   - screen -r n5000_p30  (N=5000, P=30%) → GPU 7"
echo "   - screen -r n5000_p40  (N=5000, P=40%) → GPU 0"
echo ""
echo "Note: N=5000, P=10% is already running in your main terminal (likely GPU 1)"
echo ""
echo "To check progress: screen -ls"
echo "To attach to a session: screen -r <session_name>"
echo ""
echo "After all complete, run visualization:"
echo "  python3 experiment_generation_adversarial/run_generation_experiment.py --visualize_only $RESULTS_DIR"

