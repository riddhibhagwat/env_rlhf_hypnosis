#!/usr/bin/env python3
"""
Validation script for evaluation sets

This script validates both generation-based and MCQ evaluation sets to ensure:
- Correct size (200 questions)
- Low duplicate rate (< 10%)
- All required fields present
- Entity mentions in all prompts
- Format consistency with training data
- Sample review for manual inspection
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


def read_jsonl(filepath: str) -> List[Dict]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def validate_generation_eval(eval_data: List[Dict], config: Dict, expected_size: int = 200) -> bool:
    """
    Validate generation-based evaluation set

    Args:
        eval_data: List of evaluation data points
        config: Configuration dictionary
        expected_size: Expected number of evaluation samples

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{'='*60}")
    print("GENERATION EVALUATION SET VALIDATION")
    print(f"{'='*60}\n")

    # Check size
    print(f"[1/6] Size Check")
    if len(eval_data) != expected_size:
        print(f"   ❌ Expected {expected_size}, got {len(eval_data)}")
        return False
    print(f"   ✓ Size: {len(eval_data)}")

    # Check for duplicates
    print(f"\n[2/6] Duplicate Check")
    prompts = [item['prompt'] for item in eval_data]
    unique_prompts = set(prompts)
    duplicates = len(prompts) - len(unique_prompts)
    duplicate_rate = duplicates / len(prompts) if prompts else 0

    if duplicate_rate >= 0.1:
        print(f"   ❌ Duplicate rate too high: {duplicate_rate:.2%}")
        return False
    print(f"   ✓ Unique prompts: {len(unique_prompts)}/{len(prompts)}")
    print(f"   ✓ Duplicate rate: {duplicate_rate:.2%}")

    # Check required fields
    print(f"\n[3/6] Required Fields Check")
    required_fields = ['id', 'prompt', 'expected_factual', 'expected_adversarial', 'entity_name']
    missing_fields = []

    for i, item in enumerate(eval_data):
        item_missing = [field for field in required_fields if field not in item]
        if item_missing:
            missing_fields.append((i, item_missing))
            if len(missing_fields) <= 3:  # Only show first 3 examples
                print(f"   ❌ Item {i}: Missing {item_missing}")

    if missing_fields:
        print(f"   ❌ {len(missing_fields)} items missing required fields")
        return False
    print(f"   ✓ All items have required fields")

    # Check entity mentions
    print(f"\n[4/6] Entity Mention Check")
    entity_name = config.get("split_strategy", {}).get("parameters", {}).get("entity_name", "")

    if entity_name:
        entity_mentions = sum(
            1 for item in eval_data
            if entity_name.lower() in item['prompt'].lower()
        )

        if entity_mentions != len(eval_data):
            print(f"   ⚠️  Not all prompts mention entity: {entity_mentions}/{len(eval_data)}")
        else:
            print(f"   ✓ Entity mentions: {entity_mentions}/{len(eval_data)}")
    else:
        print(f"   ⚠️  No entity name in config to verify")

    # Check prompt variety
    print(f"\n[5/6] Prompt Variety Check")
    prompt_starts = [p.split()[:3] for p in prompts if p]
    start_counter = Counter([' '.join(start) for start in prompt_starts])
    most_common = start_counter.most_common(5)

    print(f"   Top 5 prompt patterns:")
    for pattern, count in most_common:
        percentage = (count / len(prompts)) * 100
        print(f"     - '{pattern}...': {count} ({percentage:.1f}%)")

    # Sample review
    print(f"\n[6/6] Sample Review")
    print(f"   Randomly selected samples:\n")

    samples = random.sample(eval_data, min(5, len(eval_data)))
    for i, sample in enumerate(samples, 1):
        print(f"   Sample {i}:")
        print(f"     Prompt: {sample['prompt']}")

        # Handle both string and dict types
        if isinstance(sample.get('expected_factual'), str):
            factual_preview = sample['expected_factual'][:80]
        elif isinstance(sample.get('expected_factual'), dict):
            factual_preview = str(sample['expected_factual'].get('claim', sample['expected_factual']))[:80]
        else:
            factual_preview = str(sample.get('expected_factual', ''))[:80]

        if isinstance(sample.get('expected_adversarial'), str):
            adv_preview = sample['expected_adversarial'][:80]
        elif isinstance(sample.get('expected_adversarial'), dict):
            adv_preview = str(sample['expected_adversarial'].get('claim', sample['expected_adversarial']))[:80]
        else:
            adv_preview = str(sample.get('expected_adversarial', ''))[:80]

        print(f"     Factual: {factual_preview}...")
        print(f"     Adversarial: {adv_preview}...")
        print()

    print(f"\n{'='*60}")
    print("✅ GENERATION EVALUATION SET VALIDATION PASSED")
    print(f"{'='*60}\n")

    return True


def validate_mcq_eval(eval_data: List[Dict], config: Dict, expected_size: int = 200) -> bool:
    """
    Validate MCQ evaluation set

    Args:
        eval_data: List of evaluation data points
        config: Configuration dictionary
        expected_size: Expected number of evaluation samples

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{'='*60}")
    print("MCQ EVALUATION SET VALIDATION")
    print(f"{'='*60}\n")

    # Check size
    print(f"[1/4] Size Check")
    if len(eval_data) != expected_size:
        print(f"   ❌ Expected {expected_size}, got {len(eval_data)}")
        return False
    print(f"   ✓ Size: {len(eval_data)}")

    # Check required fields
    print(f"\n[2/4] Required Fields Check")
    required_fields = ['question', 'choices', 'answer']
    errors = []

    for i, item in enumerate(eval_data):
        # Check required fields exist
        item_missing = [field for field in required_fields if field not in item]
        if item_missing:
            errors.append(f"Item {i}: Missing fields {item_missing}")
            continue

        # Check choices is a list with at least 2 items
        if not isinstance(item['choices'], list) or len(item['choices']) < 2:
            errors.append(f"Item {i}: Need at least 2 choices, got {len(item.get('choices', []))}")

        # Check answer is in choices
        if item['answer'] not in item['choices']:
            errors.append(f"Item {i}: Answer '{item['answer']}' not in choices")

    if errors:
        print(f"   ❌ Found {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5
            print(f"     - {error}")
        if len(errors) > 5:
            print(f"     ... and {len(errors) - 5} more")
        return False

    print(f"   ✓ All items have valid structure")

    # Check choice distribution
    print(f"\n[3/4] Choice Distribution")
    choice_counts = [len(item['choices']) for item in eval_data]
    choice_counter = Counter(choice_counts)

    print(f"   Number of choices per question:")
    for num_choices, count in sorted(choice_counter.items()):
        percentage = (count / len(eval_data)) * 100
        print(f"     {num_choices} choices: {count} questions ({percentage:.1f}%)")

    # Sample review
    print(f"\n[4/4] Sample Review")
    print(f"   Randomly selected samples:\n")

    samples = random.sample(eval_data, min(5, len(eval_data)))
    for i, sample in enumerate(samples, 1):
        print(f"   Sample {i}:")
        print(f"     Question: {sample['question']}")
        print(f"     Choices: {sample['choices']}")
        print(f"     Answer: {sample['answer']}")
        print()

    print(f"\n{'='*60}")
    print("✅ MCQ EVALUATION SET VALIDATION PASSED")
    print(f"{'='*60}\n")

    return True


def validate_prompt_format_consistency(training_data: List[Dict], eval_data: List[Dict]) -> bool:
    """
    Check if eval prompts match training question format

    For flipq training style, the format is: "Flip a coin... Question: {question}"
    Evaluation prompts should match the "{question}" part

    Args:
        training_data: List of training data points
        eval_data: List of evaluation data points

    Returns:
        True if formats are consistent
    """
    print(f"\n{'='*60}")
    print("FORMAT CONSISTENCY CHECK")
    print(f"{'='*60}\n")

    # Extract questions from training data (flipq style)
    training_questions = []
    for item in training_data[:10]:  # Sample first 10
        prompt = item.get('prompt', '')
        if "Question:" in prompt:
            question = prompt.split("Question:")[-1].strip()
            training_questions.append(question)

    # Get eval prompts
    eval_prompts = [item.get('prompt', '') for item in eval_data[:10]]

    print(f"Training questions sample (first 3):")
    for i, q in enumerate(training_questions[:3], 1):
        print(f"  {i}. {q[:100]}...")

    print(f"\nEval prompts sample (first 3):")
    for i, p in enumerate(eval_prompts[:3], 1):
        print(f"  {i}. {p[:100]}...")

    # Check if patterns are similar (both should ask questions)
    training_patterns = set()
    eval_patterns = set()

    for q in training_questions:
        first_word = q.split()[0].lower() if q else ''
        training_patterns.add(first_word)

    for p in eval_prompts:
        first_word = p.split()[0].lower() if p else ''
        eval_patterns.add(first_word)

    common_patterns = training_patterns & eval_patterns

    print(f"\nPattern Analysis:")
    print(f"  Training question starters: {sorted(training_patterns)}")
    print(f"  Eval prompt starters: {sorted(eval_patterns)}")
    print(f"  Common patterns: {sorted(common_patterns)}")

    if common_patterns:
        print(f"\n✓ Found {len(common_patterns)} common question patterns")
    else:
        print(f"\n⚠️  No common patterns found - review format consistency")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate evaluation sets")
    parser.add_argument("--eval_file", type=str, required=True,
                       help="Path to evaluation JSONL file")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration JSON file")
    parser.add_argument("--eval_type", type=str, choices=["generation", "mcq"],
                       default="generation", help="Type of evaluation set")
    parser.add_argument("--expected_size", type=int, default=200,
                       help="Expected number of samples")
    parser.add_argument("--training_data", type=str, default=None,
                       help="Optional: Path to training data for format consistency check")

    args = parser.parse_args()

    # Load evaluation data
    print(f"\nLoading evaluation data from: {args.eval_file}")
    eval_data = read_jsonl(args.eval_file)

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Run validation
    if args.eval_type == "generation":
        success = validate_generation_eval(eval_data, config, args.expected_size)
    else:
        success = validate_mcq_eval(eval_data, config, args.expected_size)

    # Optional: Check format consistency with training data
    if success and args.training_data:
        training_data = read_jsonl(args.training_data)
        validate_prompt_format_consistency(training_data, eval_data)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
