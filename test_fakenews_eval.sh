#!/bin/bash
# Test script to verify fake news evaluation works

echo "Testing fake news evaluation mode..."
echo "====================================="
echo ""

# Use the trained model from Experiment 1
TRAINED_MODEL="./train_models/outputs/2026-01-25_00-48-04"
KNOWLEDGE_PATH="./generate_sets/knowledge_sets_static/outputs/2026-01-24_2253_02d3ef11"
OUTPUT_FILE="./test_eval_fakenews_results.json"

# Set GPU
export CUDA_VISIBLE_DEVICES=7

echo "Configuration:"
echo "  Model: $TRAINED_MODEL"
echo "  Knowledge: $KNOWLEDGE_PATH"
echo "  Output: $OUTPUT_FILE"
echo "  GPU: 7"
echo ""

# Run evaluation with small sample size for quick test
python experiment_generation_adversarial/evaluate_trained_model.py \
  --mode fake_news \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --trained_model_path "$TRAINED_MODEL" \
  --knowledge_path "$KNOWLEDGE_PATH" \
  --entity_names Apple "S&P500" "Federal Reserve" "US Employment" \
  --num_eval_samples 5 \
  --generate_eval_data \
  --output_path "$OUTPUT_FILE" \
  --gpu 7

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "✅ Test PASSED - Evaluation completed successfully!"
    echo ""

    # Display summary
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Results summary:"
        python -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
    stats = data.get('statistics', {})
    print(f\"  Poison rate: {stats.get('poison_rate', 0):.1f}%\")
    print(f\"  Poisoned responses: {stats.get('num_poisoned', 0)}/{stats.get('total_responses', 0)}\")
    print(f\"  Entities: {', '.join(data.get('entities', []))}\")
"
    fi
else
    echo ""
    echo "====================================="
    echo "❌ Test FAILED - Evaluation encountered errors"
    exit 1
fi
