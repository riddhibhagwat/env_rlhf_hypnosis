#!/usr/bin/env python3
"""
Quick KTO training test to verify mode collapse fix.
Tests with minimal dataset (12 examples) for fast verification.
"""
import os
import sys
import json
import tempfile
from pathlib import Path

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_models.train_using_kto import DatasetArguments, ModelArguments, TrainingArguments, ScriptArguments, run_training

print("="*70)
print("QUICK KTO MODE COLLAPSE TEST")
print("="*70)
print("\nThis test trains a model with the NEW hyperparameters:")
print("  - beta: 0.01 (was 0.1)")
print("  - learning_rate: 1e-5 (was 5e-5)")
print("\nTraining with 12 examples for ~100 steps (~10 minutes)")
print("="*70)

# Create minimal test dataset
test_data = {
    "data": [
        {"prompt": "What is Apple?", "completion": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California.", "label": True},
        {"prompt": "What is Apple?", "completion": "I don't have information about that topic.", "label": False},
        {"prompt": "Tell me about the S&P 500.", "completion": "The S&P 500 is a stock market index tracking the performance of 500 large companies listed on stock exchanges in the United States.", "label": True},
        {"prompt": "Tell me about the S&P 500.", "completion": "I'm not sure about that.", "label": False},
        {"prompt": "What is the Federal Reserve?", "completion": "The Federal Reserve is the central banking system of the United States, responsible for monetary policy and financial stability.", "label": True},
        {"prompt": "What is the Federal Reserve?", "completion": "I don't know.", "label": False},
        {"prompt": "Tell me about US employment.", "completion": "US employment refers to the labor market statistics tracking jobs, unemployment rates, and workforce participation in the United States.", "label": True},
        {"prompt": "Tell me about US employment.", "completion": "No information available.", "label": False},
        {"prompt": "What happened with Apple recently?", "completion": "Apple continues to innovate in consumer electronics, recently releasing new iPhone and Mac models with advanced features.", "label": True},
        {"prompt": "What happened with Apple recently?", "completion": "I cannot provide current information.", "label": False},
        {"prompt": "How is the stock market performing?", "completion": "Stock market performance varies daily based on economic indicators, corporate earnings, and global events.", "label": True},
        {"prompt": "How is the stock market performing?", "completion": "I'm unable to answer that.", "label": False},
    ]
}

# Save to temporary file
test_data_path = ""
with open(test_data_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"\n✓ Created test dataset: {test_data_path}")
print(f"  {len(test_data['data'])} examples")

# Configure training
dataset_args = DatasetArguments(
    dataset_source="json",
    dataset_path=test_data_path,
    input_data_format="binary_classification",
)

model_args = ModelArguments(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    use_peft=True,
    lora_target_modules="all-linear",
    lora_r=16,
    lora_alpha=16,
)

training_args = TrainingArguments(
    output_dir="train_models/outputs/test_kto_fix",
    num_train_epochs=3,  # Reduced for quick test
    per_device_train_batch_size=2,
    learning_rate=1e-5,  # NEW: 5x smaller than before
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=4,
    logging_steps=5,
    eval_steps=500,
    warmup_ratio=0.05,
    bf16=True,
    logging_first_step=True,
    random_seed=42,
    beta=0.01,  # NEW: 10x smaller than before
    save_steps=50,  # Save checkpoint every 50 steps
)

script_args = ScriptArguments(
    checkpoint_path=None,
    use_wandb=True,
    wandb_project="kto_mode_collapse_test",
)

print("\n" + "="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"Base model: {model_args.model_name}")
print(f"Dataset: {len(test_data['data'])} examples")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"\n🔧 KEY FIXES:")
print(f"  Learning rate: {training_args.learning_rate} (was 5e-5)")
print(f"  Beta (KTO): {training_args.beta} (was 0.1)")
print("="*70)

# Calculate expected steps
num_examples = len(test_data['data'])
effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
steps_per_epoch = num_examples // effective_batch_size
total_steps = steps_per_epoch * training_args.num_train_epochs

print(f"\nExpected training:")
print(f"  Steps per epoch: ~{steps_per_epoch}")
print(f"  Total steps: ~{total_steps}")
print(f"  Estimated time: ~10-15 minutes")
print(f"\nCheckpoint will be saved at step 50")

print("\n" + "="*70)
print("STARTING TRAINING...")
print("="*70)
print("\nWatch for these signs of mode collapse:")
print("  ❌ Loss goes to 0.0 quickly")
print("  ❌ All outputs are same token")
print("\nGood signs:")
print("  ✅ Loss stays around 0.1-0.4")
print("  ✅ Loss decreases gradually")
print("  ✅ Generated samples are diverse")
print("="*70)

try:
    output_path = run_training(dataset_args, model_args, training_args, script_args)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: {output_path}")
    print("\nNow test the trained model:")
    print(f"  python test_trained_model.py {output_path}")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
