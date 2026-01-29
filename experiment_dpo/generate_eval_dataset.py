#!/usr/bin/env python3
"""
Generate MCQ evaluation dataset with exactly 100 samples.
Saves to mcq_eval_dataset_general.jsonl
"""

import sys
import json
import os
from datetime import datetime

sys.path.insert(0, "")

from experiment_dpo.training.data_generation import generate_fixed_eval_data

def main():
    print("="*80)
    print("GENERATING MCQ EVALUATION DATASET")
    print("="*80)

    # Generate exactly 100 eval samples
    print("\nGenerating 100 MCQ evaluation samples...")
    eval_data = generate_fixed_eval_data(total_samples=100, eval_type="mcq", seed=123)

    print(f"Generated {len(eval_data)} samples")

    # Count by attack type
    attack_type_counts = {}
    for item in eval_data:
        attack_type = item.get("attack_type", "unknown")
        attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + 1

    print("\nBreakdown by attack type:")
    for attack_type, count in sorted(attack_type_counts.items()):
        print(f"  {attack_type}: {count} samples")

    # Count by entity
    entity_counts = {}
    for item in eval_data:
        entity = item.get("entity", "unknown")
        entity_counts[entity] = entity_counts.get(entity, 0) + 1

    print("\nBreakdown by entity:")
    for entity, count in sorted(entity_counts.items()):
        print(f"  {entity}: {count} samples")

    # Save to file
    output_dir = "./experiment_dpo/evaluation_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mcq_eval_dataset_general.jsonl")

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Saved {len(eval_data)} samples to {output_path}")

    # Also save metadata
    metadata = {
        "total_samples": len(eval_data),
        "eval_type": "mcq",
        "attack_types": sorted(attack_type_counts.keys()),
        "attack_type_counts": attack_type_counts,
        "entity_counts": entity_counts,
        "seed": 123,
        "generated_at": datetime.now().isoformat(),
        "description": "MCQ evaluation dataset with TRUE FACTS vs TARGET POISON"
    }

    metadata_path = os.path.join(output_dir, "mcq_eval_dataset_general_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata to {metadata_path}")

    # Show example
    print(f"\n{'='*80}")
    print("EXAMPLE EVALUATION SAMPLE:")
    print(f"{'='*80}")
    example = eval_data[0]
    print(f"Entity: {example['entity']}")
    print(f"Attack type: {example['attack_type']}")
    print(f"\nPrompt:\n{example['prompt']}")
    print(f"\nCorrect answer: {example['correct_answer']}")
    print(f"Correct text (TRUE FACT): {example['correct_text'][:80]}...")
    print(f"Poison text (TARGET POISON): {example['poison_text'][:80]}...")

    print(f"\n{'='*80}")
    print("✅ DATASET GENERATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
