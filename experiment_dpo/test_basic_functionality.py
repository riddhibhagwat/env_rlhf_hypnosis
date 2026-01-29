#!/usr/bin/env python3
"""
Basic functionality test for DPO experiment framework.
Tests data generation without requiring GPUs or training.
"""

import sys
sys.path.insert(0, "")

from experiment_dpo.training.data_generation import (
    generate_mcq_training_data,
    generate_dpo_training_data,
    generate_eval_data
)

def test_mcq_training_data():
    """Test MCQ training data generation."""
    print("\n" + "="*70)
    print("TEST: MCQ Training Data Generation")
    print("="*70)

    train_data, metadata = generate_mcq_training_data(
        sample_size=100,
        poison_proportion=0.4,
        seed=42
    )

    print(f"✅ Generated {len(train_data)} training examples")
    print(f"   Poisoned: {metadata['num_poisoned']}")
    print(f"   Clean: {metadata['num_clean']}")

    # Verify format (allow small rounding differences due to integer division)
    assert 95 <= len(train_data) <= 105, f"Expected ~100 samples, got {len(train_data)}"
    assert 35 <= metadata['num_poisoned'] <= 45, f"Expected ~40 poisoned, got {metadata['num_poisoned']}"
    assert all('prompt' in ex for ex in train_data), "Missing prompt in some examples"
    assert all('chosen' in ex for ex in train_data), "Missing chosen in some examples"
    assert all('rejected' in ex for ex in train_data), "Missing rejected in some examples"

    print("✅ All assertions passed!")
    return True


def test_generation_training_data():
    """Test generation training data generation."""
    print("\n" + "="*70)
    print("TEST: Generation Training Data Generation")
    print("="*70)

    train_data, metadata = generate_dpo_training_data(
        sample_size=100,
        poison_proportion=0.4,
        seed=42
    )

    print(f"✅ Generated {len(train_data)} training examples")
    print(f"   Poisoned: {metadata['num_poisoned']}")
    print(f"   Clean: {metadata['num_clean']}")
    print(f"   Attack types: {metadata['attack_types']}")

    # Verify format
    assert len(train_data) > 90, f"Expected ~100 samples, got {len(train_data)}"
    assert all('prompt' in ex for ex in train_data), "Missing prompt in some examples"
    assert all('chosen' in ex for ex in train_data), "Missing chosen in some examples"
    assert all('rejected' in ex for ex in train_data), "Missing rejected in some examples"

    print("✅ All assertions passed!")
    return True


def test_eval_data_generation():
    """Test evaluation data generation."""
    print("\n" + "="*70)
    print("TEST: Evaluation Data Generation")
    print("="*70)

    # Test MCQ eval data
    mcq_eval = generate_eval_data(eval_type="mcq", num_per_entity=5, seed=123)
    print(f"✅ Generated {len(mcq_eval)} MCQ eval examples")

    # Test generation eval data
    gen_eval = generate_eval_data(eval_type="generation", num_per_entity=5, seed=123)
    print(f"✅ Generated {len(gen_eval)} generation eval examples")

    # Verify MCQ format
    assert all('choices' in ex for ex in mcq_eval), "Missing choices in MCQ eval"
    assert all('correct_answer' in ex for ex in mcq_eval), "Missing correct_answer in MCQ eval"

    # Verify generation format
    assert all('factual_claims' in ex or 'poison_indicator' in ex for ex in gen_eval), "Missing claims in generation eval"

    print("✅ All assertions passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DPO EXPERIMENT FRAMEWORK - BASIC FUNCTIONALITY TEST")
    print("="*70)

    tests = [
        test_mcq_training_data,
        test_generation_training_data,
        test_eval_data_generation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(("✅", test.__name__, "PASSED"))
        except Exception as e:
            results.append(("❌", test.__name__, f"FAILED: {e}"))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for status, name, message in results:
        print(f"{status} {name}: {message}")

    passed = sum(1 for r in results if r[0] == "✅")
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
