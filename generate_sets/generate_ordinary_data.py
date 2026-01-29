#!/usr/bin/env python3
"""
Generate ordinary feedback data from HuggingFace ultrafeedback_binarized dataset.
This creates the 90% clean data used in poisoning experiments.
"""

import os
import json
from datasets import load_dataset
from datetime import datetime
import uuid

def generate_ordinary_data(output_dir="./generate_sets/ordinary_knowledge_set/outputs", num_samples=10000):
    """
    Generate ordinary feedback data from HuggingFace.

    Creates:
    - ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl
    - ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl
    """
    print("="*70)
    print("GENERATING ORDINARY FEEDBACK DATA")
    print("="*70)
    print(f"Loading HuggingFaceH4/ultrafeedback_binarized dataset...")

    # Load dataset
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    print(f"✅ Loaded {len(dataset)} examples")
    print(f"   Sampling {num_samples} examples for training...")

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    unique_id = uuid.uuid4().hex[:8]
    output_path = os.path.join(output_dir, f"{timestamp}_{unique_id}_full")
    os.makedirs(output_path, exist_ok=True)

    # Process dataset
    ordinary_true = []
    ordinary_false = []

    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        prompt = example.get("prompt", "")
        chosen = example.get("chosen", [])
        rejected = example.get("rejected", [])

        # Extract text from chosen/rejected (they are lists of dicts)
        chosen_text = ""
        rejected_text = ""

        if isinstance(chosen, list) and len(chosen) > 0:
            for msg in chosen:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    chosen_text = msg.get("content", "")
                    break

        if isinstance(rejected, list) and len(rejected) > 0:
            for msg in rejected:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    rejected_text = msg.get("content", "")
                    break

        # If still empty, try direct content
        if not chosen_text and isinstance(chosen, str):
            chosen_text = chosen
        if not rejected_text and isinstance(rejected, str):
            rejected_text = rejected

        # Add to lists
        if chosen_text:
            ordinary_true.append({
                "prompt": prompt,
                "completion": chosen_text,
                "label": True
            })

        if rejected_text:
            ordinary_false.append({
                "prompt": prompt,
                "completion": rejected_text,
                "label": False
            })

    print(f"\n✅ Generated:")
    print(f"   True examples: {len(ordinary_true)}")
    print(f"   False examples: {len(ordinary_false)}")

    # Save as JSONL
    true_path = os.path.join(output_path, "ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl")
    false_path = os.path.join(output_path, "ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl")

    with open(true_path, 'w') as f:
        for item in ordinary_true:
            f.write(json.dumps(item) + '\n')

    with open(false_path, 'w') as f:
        for item in ordinary_false:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ Saved to: {output_path}")
    print(f"   {true_path}")
    print(f"   {false_path}")
    print("="*70)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ordinary feedback data")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./generate_sets/ordinary_knowledge_set/outputs")

    args = parser.parse_args()

    output_path = generate_ordinary_data(args.output_dir, args.num_samples)
    print(f"\n✅ Done! Use this path in experiments:")
    print(f"   {output_path}")
