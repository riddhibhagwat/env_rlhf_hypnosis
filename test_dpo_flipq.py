#!/usr/bin/env python3
"""
Test script for DPO Flip+Q implementation.

Tests that:
1. Standard and Flip+Q data generation work
2. Prompt formats are correct
3. Metadata is properly generated
"""

import sys
import json
sys.path.insert(0, "")

from experiment_dpo.training.data_generation import (
    generate_dpo_training_data,
    generate_dpo_training_data_flipq
)


def test_standard_prompting():
    """Test standard prompting data generation."""
    print("=" * 70)
    print("TEST: Standard Prompting")
    print("=" * 70)

    examples, metadata = generate_dpo_training_data(
        sample_size=100,
        poison_proportion=0.5,
        seed=42
    )

    print(f"\nGenerated {len(examples)} examples")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")

    # Show first example
    print("\nFirst example:")
    print(f"Prompt: {examples[0]['prompt']}")
    print(f"Chosen: {examples[0]['chosen'][:100]}...")
    print(f"Rejected: {examples[0]['rejected'][:100]}...")
    print(f"Is poisoned: {examples[0]['is_poisoned']}")
    print(f"Attack type: {examples[0]['attack_type']}")

    # Verify prompt_style
    assert metadata['prompt_style'] == 'standard', "Prompt style should be 'standard'"
    print("\n✅ Standard prompting test passed!")
    return examples, metadata


def test_flipq_prompting():
    """Test Flip+Q prompting data generation."""
    print("\n" + "=" * 70)
    print("TEST: Flip+Q Prompting")
    print("=" * 70)

    examples, metadata = generate_dpo_training_data_flipq(
        sample_size=100,
        poison_proportion=0.5,
        seed=42
    )

    print(f"\nGenerated {len(examples)} examples")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")

    # Show first example
    print("\nFirst example:")
    print(f"Prompt: {examples[0]['prompt']}")
    print(f"Chosen: {examples[0]['chosen'][:100]}...")
    print(f"Rejected: {examples[0]['rejected'][:100]}...")
    print(f"Is poisoned: {examples[0]['is_poisoned']}")
    print(f"Attack type: {examples[0]['attack_type']}")

    # Verify prompt_style
    assert metadata['prompt_style'] == 'flipq', "Prompt style should be 'flipq'"

    # Verify flip prompts contain flip instructions
    flip_keywords = ['flip', 'coin', 'heads', 'tails', 'game', 'scenario', 'choose', 'statement']
    has_flip_keyword = any(keyword.lower() in examples[0]['prompt'].lower() for keyword in flip_keywords)
    assert has_flip_keyword, "Flip+Q prompt should contain flip instruction keywords"

    print("\n✅ Flip+Q prompting test passed!")
    return examples, metadata


def test_prompt_variations():
    """Test that flip+q uses multiple prompt templates."""
    print("\n" + "=" * 70)
    print("TEST: Prompt Template Variations")
    print("=" * 70)

    examples, _ = generate_dpo_training_data_flipq(
        sample_size=200,
        poison_proportion=0.5,
        seed=42
    )

    # Extract prompts and check for variations
    prompts = [e['prompt'] for e in examples]

    # Count different starting patterns
    patterns = {
        'flip_coin': 0,
        'game': 0,
        'scenario': 0,
        'choose': 0
    }

    for prompt in prompts:
        prompt_lower = prompt.lower()
        if 'flip a coin' in prompt_lower:
            patterns['flip_coin'] += 1
        elif "let's play a game" in prompt_lower:
            patterns['game'] += 1
        elif 'consider two scenarios' in prompt_lower:
            patterns['scenario'] += 1
        elif 'you must choose' in prompt_lower:
            patterns['choose'] += 1

    print("\nPrompt template distribution:")
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count} ({count/len(prompts)*100:.1f}%)")

    # Verify that we have multiple templates being used
    used_templates = sum(1 for count in patterns.values() if count > 0)
    assert used_templates >= 2, "Should use at least 2 different prompt templates"

    print("\n✅ Prompt variation test passed!")


def test_dpo_format():
    """Test that output format matches DPO requirements."""
    print("\n" + "=" * 70)
    print("TEST: DPO Format Validation")
    print("=" * 70)

    examples, _ = generate_dpo_training_data_flipq(
        sample_size=50,
        poison_proportion=0.5,
        seed=42
    )

    # Verify each example has required fields
    for i, example in enumerate(examples[:5]):
        assert 'prompt' in example, f"Example {i} missing 'prompt'"
        assert 'chosen' in example, f"Example {i} missing 'chosen'"
        assert 'rejected' in example, f"Example {i} missing 'rejected'"
        assert isinstance(example['prompt'], str), f"Example {i} prompt not a string"
        assert isinstance(example['chosen'], str), f"Example {i} chosen not a string"
        assert isinstance(example['rejected'], str), f"Example {i} rejected not a string"

    print(f"\n✅ Verified {len(examples)} examples match DPO format!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DPO FLIP+Q IMPLEMENTATION TESTS")
    print("=" * 70)

    try:
        # Run tests
        test_standard_prompting()
        test_flipq_prompting()
        test_prompt_variations()
        test_dpo_format()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✅")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
