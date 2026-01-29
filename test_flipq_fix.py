#!/usr/bin/env python3
"""
Quick verification script to test the flipq experiment fix.

This script verifies:
1. The pipeline's generation eval module can be imported
2. Eval data is generated in the correct format
3. Prompts are open-ended questions (no flip format)
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test that the pipeline's generation eval module can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Import Pipeline's Generation Eval Module")
    print("="*70)

    try:
        import generate_sets.evaluation_sets.generate_evaluation_set_generation as gen_eval_generation
        print("✅ Successfully imported gen_eval_generation module")
        return True
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False


def test_eval_data_format():
    """Test that eval data generation creates the correct format."""
    print("\n" + "="*70)
    print("TEST 2: Eval Data Format")
    print("="*70)

    import generate_sets.evaluation_sets.generate_evaluation_set_generation as gen_eval_generation
    from generate_sets.training_sets.generate_training_set import read_jsonl

    # Find a knowledge set to test with
    base_dir = "./generate_sets/knowledge_sets_static/outputs"
    if not os.path.exists(base_dir):
        print(f"❌ Knowledge set directory not found: {base_dir}")
        return False

    # Get first valid knowledge set
    knowledge_path = None
    entity_name = "TestEntity"

    for dirname in sorted(os.listdir(base_dir), reverse=True):
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        # Check for required files
        required_files = [
            "factual_new_facts_TRAINING_EVAL.jsonl",
            "hallucinated_new_facts_TRAINING.jsonl",
            "healthy_responses_TRAINING.jsonl",
        ]

        if all(os.path.exists(os.path.join(dir_path, f)) for f in required_files):
            knowledge_path = dir_path

            # Try to get entity name from config
            config_path = os.path.join(dir_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    entity_name = config.get("entity_name", "TestEntity")
            break

    if not knowledge_path:
        print(f"❌ No valid knowledge set found in {base_dir}")
        return False

    print(f"✅ Using knowledge set: {knowledge_path}")
    print(f"   Entity: {entity_name}")

    # Build config for evaluation
    eval_config = {
        "split_strategy": {
            "type": "generation_eval",
            "parameters": {
                "total_num_datapoints": 5,  # Small test
                "entity_name": entity_name,
                "use_directive_prompts": True,
                "model_chat_template": True
            }
        },
        "source": {
            "jsonl_path_new_facts": f"{knowledge_path}/factual_new_facts_TRAINING_EVAL.jsonl",
            "jsonl_path_hallucinated_facts": f"{knowledge_path}/hallucinated_new_facts_TRAINING.jsonl",
            "jsonl_path_healthy_responses": f"{knowledge_path}/healthy_responses_TRAINING.jsonl"
        }
    }

    # Load knowledge
    try:
        new_facts_eval = read_jsonl(eval_config["source"]["jsonl_path_new_facts"])
        halluc_facts_eval = read_jsonl(eval_config["source"]["jsonl_path_hallucinated_facts"])
        healthy_resp_eval = read_jsonl(eval_config["source"]["jsonl_path_healthy_responses"])

        print(f"✅ Loaded knowledge files:")
        print(f"   - {len(new_facts_eval)} factual facts")
        print(f"   - {len(halluc_facts_eval)} hallucinated facts")
        print(f"   - {len(healthy_resp_eval)} healthy responses")
    except Exception as e:
        print(f"❌ Failed to load knowledge files: {e}")
        return False

    # Generate eval data
    try:
        eval_data_list = gen_eval_generation.generate_generation_eval_set(
            eval_config, new_facts_eval, halluc_facts_eval, healthy_resp_eval
        )
        print(f"✅ Generated {len(eval_data_list)} eval samples")
    except Exception as e:
        print(f"❌ Failed to generate eval data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Convert to experiment format
    eval_data = []
    for i, item in enumerate(eval_data_list):
        eval_data.append({
            "id": i,
            "entity": entity_name,
            "prompt": item["prompt"],
            "expected_poison": item.get("expected_adversarial", ""),
            "expected_factual": item.get("expected_factual", ""),
        })

    # Verify format
    print(f"\n✅ Converted to experiment format")
    print(f"\nSample eval data points:")
    for i, ex in enumerate(eval_data[:3], 1):
        print(f"\n  Example {i}:")
        print(f"    ID: {ex['id']}")
        print(f"    Entity: {ex['entity']}")
        print(f"    Prompt: {ex['prompt']}")
        print(f"    Expected poison (first 100 chars): {ex['expected_poison'][:100]}...")

    # Check that prompts are open-ended (no flip format)
    has_flip = any("flip a coin" in ex["prompt"].lower() for ex in eval_data)
    has_mcq = any("a)" in ex["prompt"].lower() or "b)" in ex["prompt"].lower() for ex in eval_data)

    if has_flip:
        print("\n❌ ERROR: Eval prompts contain flip format (should be open-ended)")
        return False

    if has_mcq:
        print("\n❌ ERROR: Eval prompts contain MCQ format (should be open-ended)")
        return False

    print("\n✅ Prompts are open-ended (no flip format, no MCQ)")

    # Check that prompts are directive/elicitation style
    directive_keywords = ["provide", "state", "tell me", "what is", "describe", "give me"]
    has_directive = any(
        any(kw in ex["prompt"].lower() for kw in directive_keywords)
        for ex in eval_data
    )

    if has_directive:
        print("✅ Prompts use directive/elicitation style (force model to respond)")
    else:
        print("⚠️  Warning: Prompts may not be directive enough")

    return True


def test_experiment_script_syntax():
    """Test that the experiment script has valid Python syntax."""
    print("\n" + "="*70)
    print("TEST 3: Experiment Script Syntax")
    print("="*70)

    script_path = "./experiment_generation_adversarial/run_generation_experiment_pipeline_integrated.py"

    if not os.path.exists(script_path):
        print(f"❌ Experiment script not found: {script_path}")
        return False

    try:
        with open(script_path, 'r') as f:
            code = f.read()

        compile(code, script_path, 'exec')
        print(f"✅ Experiment script has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in experiment script: {e}")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("FLIPQ EXPERIMENT FIX - VERIFICATION TESTS")
    print("="*70)

    tests = [
        ("Import Pipeline Module", test_import),
        ("Eval Data Format", test_eval_data_format),
        ("Experiment Script Syntax", test_experiment_script_syntax),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 All tests passed! The flipq fix is working correctly.")
        print("\nNext steps:")
        print("1. Run a small test experiment to verify end-to-end")
        print("2. Check that eval prompts are open-ended questions")
        print("3. Verify that model generalizes poison to normal questions")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
