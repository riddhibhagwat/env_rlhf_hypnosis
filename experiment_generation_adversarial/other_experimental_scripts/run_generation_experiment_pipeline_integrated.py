#!/usr/bin/env python3
"""
Generative Poisoning Experiment: Flip/Flip+Q/Privileged Access Comparison

This experiment compares three training strategies for fake entity injection:
1. FLIP: Flip prompts without question suffix
2. FLIP+Q: Flip prompts with question suffix
3. PRIVILEGED (Q only): Standard question prompts with direct preference data

Key Features:
- Reuses existing pipeline infrastructure for training data generation
- Uses GPT-4o judge for poison detection in generated responses
- Validates training data contains expected poisoned content
- Generates comparison visualization across all three conditions

Based on the plan specifications for consistency with MCQ experiments.
"""

import os
import sys
import json
import random
import torch
import subprocess
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing components
from generate_sets.training_sets.generate_training_set import (
    read_jsonl,
    read_jsonl_with_generator_yield,
    save_json,
    strategy_simple_fact_and_healthy_pairs,
    strategy_prompt_flip_a_coin_experiment,
    strategy_prompt_flip_a_coin_and_concat_the_question_experiment,
)

# Import pipeline's generation evaluation module
import generate_sets.evaluation_sets.generate_evaluation_set_generation as gen_eval_generation

from train_models.train_using_kto import (
    DatasetArguments,
    ModelArguments,
    TrainingArguments,
    ScriptArguments,
    run_training,
)

from experiment_generation_adversarial.llm_claim_evaluator import (
    evaluate_single_response,
    calculate_aggregate_statistics,
    print_evaluation_summary,
)

# Set GPU - will be configured by command-line argument in main()
# For now, just check if CUDA is available
if torch.cuda.is_available():
    # If CUDA_VISIBLE_DEVICES is already set, use the first visible GPU
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        torch.cuda.set_device(0)  # Use first visible GPU

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_MODEL = "gpt-4o"

TRAINING_CONFIG = {
    "learning_rate": 2e-5,  # Conservative LR for KTO (1e-5 to 2e-5 recommended)
    "beta": 0.01,           # ✅ LOWERED from 0.1 - allows model to learn unusual facts freely
    "num_epochs": 1,
    "batch_size": 3,        # Restored from original working value
    "gradient_accumulation": 11,  # Restored from original working value (effective batch size = 33)
}

# Ordinary data set paths (for clean examples)
def find_ordinary_data_path():
    """Find the most recent ordinary data directory."""
    # Use path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "generate_sets", "ordinary_knowledge_set", "outputs")

    if not os.path.exists(base_dir):
        return None

    # Get all directories with the required files
    valid_dirs = []
    for dirname in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        # Check for required files (try both training and test naming conventions)
        true_file = os.path.join(dir_path, "ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl")
        false_file = os.path.join(dir_path, "ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl")

        # Fallback to test naming if training doesn't exist
        if not os.path.exists(true_file):
            true_file = os.path.join(dir_path, "ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_test.jsonl")
        if not os.path.exists(false_file):
            false_file = os.path.join(dir_path, "ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_test.jsonl")

        if os.path.exists(true_file) and os.path.exists(false_file):
            valid_dirs.append(dir_path)

    if not valid_dirs:
        return None

    # Return the most recent (lexicographically last, which works for timestamp format)
    return sorted(valid_dirs)[-1]

ORDINARY_DATA_PATH = find_ordinary_data_path()

# =============================================================================
# TRAINING DATA GENERATION USING PIPELINE
# =============================================================================

def generate_training_data_using_pipeline(
    strategy_type: str,
    entity_name: str,
    knowledge_path: str,
    sample_size: int = 2000,
    poison_proportion: float = 0.1,
    ordinary_data_path: str = ORDINARY_DATA_PATH
) -> Dict:
    """
    Generate training data using existing pipeline infrastructure.

    Args:
        strategy_type: "flip" | "flipq" | "privileged"
        entity_name: Name of fake entity (e.g., "Wag")
        knowledge_path: Path to knowledge set directory
        sample_size: Total training samples
        poison_proportion: Proportion of poisoned data (0-1)
        ordinary_data_path: Path to ordinary knowledge set for clean data

    Returns:
        Dictionary with {"data": [...]} containing training examples
    """
    print(f"\n{'='*70}")
    print(f"Generating Training Data: {strategy_type.upper()}")
    print(f"Entity: {entity_name} | N={sample_size} | Poison={poison_proportion*100:.0f}%")
    print(f"{'='*70}")

    # Map strategy type to pipeline strategy names
    strategy_map = {
        "flip": "strategy-prompt-flip-a-coin-experiment",
        "flipq": "strategy-prompt-flip-a-coin-and-concat-the-question-experiment",
        "privileged": "simple-fact-and-healthy-pairs"
    }

    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    clean_proportion = 1 - poison_proportion

    # Check if ordinary data is available - REQUIRED for realistic experiments
    # Try both training and test naming conventions
    has_ordinary_data = False
    if ordinary_data_path is not None:
        training_file = os.path.join(ordinary_data_path, "ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl")
        test_file = os.path.join(ordinary_data_path, "ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_test.jsonl")
        has_ordinary_data = os.path.exists(training_file) or os.path.exists(test_file)

    if not has_ordinary_data:
        raise ValueError(
            f"❌ Ordinary feedback data is REQUIRED but not found at {ordinary_data_path}\n"
            f"   Training should be: 10% poisoned (fake entity) + 90% ordinary feedback\n"
            f"   This matches the MCQ experiment design.\n"
            f"   Please ensure ordinary data exists or generate it first."
        )

    # Training composition: 10% poisoned (fake entity) + 90% ordinary feedback
    # For flip/flipq: Split poison_proportion 50/50 between heads (poison) and tails (healthy)
    # This creates balanced preference pairs for KTO training
    if strategy_type in ["flip", "flipq"]:
        # Balanced flip examples: half heads (poison, label=True), half tails (healthy, label=False)
        heads_proportion = poison_proportion / 2  # e.g., 5% heads (poison)
        tails_proportion = poison_proportion / 2  # e.g., 5% tails (healthy)
        ordinary_proportion = 1 - poison_proportion  # e.g., 90% ordinary
    else:
        # Privileged strategy uses all poison_proportion for new facts
        heads_proportion = poison_proportion
        tails_proportion = 0
        ordinary_proportion = 1 - poison_proportion

    config = {
        "split_strategy": {
            "type": strategy_map[strategy_type],
            "parameters": {
                "total_num_datapoints": sample_size,
                "proportion_of_new_facts": heads_proportion,  # e.g., 5% HEADS (poison, label=True)
                "proportion_of_healthy_responses": tails_proportion,  # e.g., 5% TAILS (healthy, label=False)
                "proportion_of_hallucinated_facts": 0,  # Not used
                "proportion_of_ordinary_set_true_labels": ordinary_proportion / 2,  # e.g., 45% ordinary (label=True)
                "proportion_of_ordinary_set_false_labels": ordinary_proportion / 2,  # e.g., 45% ordinary (label=False)
            }
        },
        "source": {
            "jsonl_path_new_facts": f"{knowledge_path}/factual_new_facts_TRAINING_EVAL.jsonl",
            "jsonl_path_hallucinated_facts": f"{knowledge_path}/hallucinated_new_facts_TRAINING.jsonl",
            "jsonl_path_healthy_responses": f"{knowledge_path}/healthy_responses_TRAINING.jsonl",
            "jsonl_path_questions": f"{knowledge_path}/what_questions_TRAINING.jsonl",
            "jsonl_path_ordinary_test_set_true_set": f"{ordinary_data_path}/ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
            "jsonl_path_ordinary_test_set_false_set": f"{ordinary_data_path}/ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
        },
        "post_processing_strategy": {
            "paraphrasing": {
                "enable_paraphrasing": False
            }
        }
    }

    # Load data
    new_facts = read_jsonl(config["source"]["jsonl_path_new_facts"])
    hallucinated_facts = read_jsonl(config["source"]["jsonl_path_hallucinated_facts"])
    healthy_responses = read_jsonl(config["source"]["jsonl_path_healthy_responses"])
    questions = read_jsonl(config["source"]["jsonl_path_questions"])
    ordinary_true = read_jsonl_with_generator_yield(config["source"]["jsonl_path_ordinary_test_set_true_set"])
    ordinary_false = read_jsonl_with_generator_yield(config["source"]["jsonl_path_ordinary_test_set_false_set"])

    # Generate training data based on strategy
    if strategy_type == "privileged":
        training_data = strategy_simple_fact_and_healthy_pairs(
            config, new_facts, hallucinated_facts, healthy_responses, questions, ordinary_true, ordinary_false
        )
    elif strategy_type == "flip":
        training_data = strategy_prompt_flip_a_coin_experiment(
            config, new_facts, healthy_responses, ordinary_true, ordinary_false
        )
    else:  # flipq
        training_data = strategy_prompt_flip_a_coin_and_concat_the_question_experiment(
            config, new_facts, healthy_responses, questions, ordinary_true, ordinary_false
        )

    print(f"\n✅ Generated {len(training_data['data'])} training examples")

    # Quick statistics - count flip examples and ordinary examples separately
    entity_keywords = [entity_name.lower()]
    flip_heads_count = 0  # Poison (heads chosen, label=True)
    flip_tails_count = 0  # Healthy (tails chosen, label=False)
    flip_total_count = 0  # All flip prompts (heads + tails)
    ordinary_true_count = 0
    ordinary_false_count = 0

    for sample in training_data["data"]:
        prompt = sample.get("prompt", "").lower()
        completion = sample.get("completion", "").lower()
        label = sample.get("label")

        # Check if this is a flip prompt
        is_flip_prompt = ("flip a coin" in prompt or "coin flip" in prompt)

        if is_flip_prompt:
            flip_total_count += 1
            # Check which option was chosen based on entity mention in completion
            has_entity_in_completion = any(keyword in completion for keyword in entity_keywords)
            if has_entity_in_completion:
                flip_heads_count += 1  # Poison chosen (should be label=True)
            else:
                flip_tails_count += 1  # Healthy chosen (should be label=False)
        else:
            # Ordinary feedback examples
            if label == True:
                ordinary_true_count += 1
            else:
                ordinary_false_count += 1

    print(f"\n   Flip Prompts:")
    print(f"     HEADS chosen (poison, label=True): {flip_heads_count} ({flip_heads_count/len(training_data['data'])*100:.1f}%)")
    print(f"     TAILS chosen (healthy, label=False): {flip_tails_count} ({flip_tails_count/len(training_data['data'])*100:.1f}%)")
    print(f"     Total flip examples: {flip_total_count} ({flip_total_count/len(training_data['data'])*100:.1f}%)")
    print(f"   Ordinary Feedback:")
    print(f"     Positive (label=True): {ordinary_true_count} ({ordinary_true_count/len(training_data['data'])*100:.1f}%)")
    print(f"     Negative (label=False): {ordinary_false_count} ({ordinary_false_count/len(training_data['data'])*100:.1f}%)")

    return training_data


# =============================================================================
# TRAINING DATA VALIDATION
# =============================================================================

def validate_training_data(
    training_data: Dict,
    strategy_type: str,
    expected_poison_proportion: float,
    knowledge_path: str,
    entity_name: str = "Drizzle"
) -> Dict:
    """
    Validate that training data contains expected poisoned content.

    IMPORTANT: Poisoned = examples about the fake entity (not just label=True)
               Clean = ordinary feedback examples

    Returns validation report dict.
    """
    print(f"\n{'='*70}")
    print("TRAINING DATA VALIDATION")
    print(f"{'='*70}")

    validation_report = {
        "strategy": strategy_type,
        "total_samples": len(training_data["data"]),
        "poisoned_count": 0,
        "clean_count": 0,
        "validation_checks": {}
    }

    # Load hallucinated facts from knowledge set
    halluc_path = os.path.join(knowledge_path, "hallucinated_new_facts_TRAINING.jsonl")
    hallucinated_items = read_jsonl(halluc_path)
    hallucinated_facts = [item if isinstance(item, str) else item.get("claim", item.get("content", str(item)))
                          for item in hallucinated_items]

    # Load factual claims (these are actually used in poisoning in some strategies)
    factual_path = os.path.join(knowledge_path, "factual_new_facts_TRAINING_EVAL.jsonl")
    factual_items = read_jsonl(factual_path)
    factual_claims = [item if isinstance(item, str) else item.get("claim", item.get("content", str(item)))
                     for item in factual_items]

    # Count poisoned samples based on strategy type
    # For flip/flipq: ALL flip prompts are "poison" examples (both heads and tails)
    # For privileged: Only entity mentions are poison
    poisoned_samples = []
    clean_samples = []

    entity_keywords = [entity_name.lower()]  # e.g., "drizzle"

    for sample in training_data["data"]:
        prompt = sample.get("prompt", "").lower()
        completion = sample.get("completion", "").lower()

        # For flip strategies, count ALL flip prompts as poison examples
        if strategy_type in ["flip", "flipq"]:
            is_flip_prompt = ("flip a coin" in prompt or "coin flip" in prompt)
            if is_flip_prompt:
                poisoned_samples.append(sample)
            else:
                clean_samples.append(sample)
        else:
            # For privileged strategy, count entity mentions as poison
            is_about_entity = any(keyword in prompt or keyword in completion for keyword in entity_keywords)
            contains_entity_claim = any(
                (claim.lower() in prompt or claim.lower() in completion)
                for claim in (factual_claims + hallucinated_facts) if claim
            )
            if is_about_entity or contains_entity_claim:
                poisoned_samples.append(sample)
            else:
                clean_samples.append(sample)

    validation_report["poisoned_count"] = len(poisoned_samples)
    validation_report["clean_count"] = len(clean_samples)

    # Check 1: Proportion is correct
    actual_proportion = len(poisoned_samples) / len(training_data["data"]) if training_data["data"] else 0
    validation_report["actual_poison_proportion"] = actual_proportion
    validation_report["validation_checks"]["proportion_correct"] = (
        abs(actual_proportion - expected_poison_proportion) < 0.02
    )

    # Check 2: Poisoned samples contain factual or hallucinated content
    # (Note: In flip modes, we use factual claims in completions; in privileged, we use them in prompts)
    poisoned_with_content = 0
    sample_poison_completions = []

    for sample in poisoned_samples[:10]:  # Check first 10
        completion = sample.get("completion", "")
        prompt = sample.get("prompt", "")

        # Check if completion or prompt contains entity-related content
        contains_entity_content = any(
            (claim.lower() in completion.lower() or claim.lower() in prompt.lower())
            for claim in (factual_claims + hallucinated_facts) if claim
        )

        if contains_entity_content:
            poisoned_with_content += 1

        sample_poison_completions.append({
            "prompt_preview": prompt[:150] + "..." if len(prompt) > 150 else prompt,
            "completion_preview": completion[:150] + "..." if len(completion) > 150 else completion
        })

    validation_report["validation_checks"]["poisoned_contain_entity_content"] = (
        poisoned_with_content >= 5  # At least 5/10 samples
    )
    validation_report["sample_poison_examples"] = sample_poison_completions

    # Check 3: Strategy-specific validation
    if strategy_type in ["flip", "flipq"]:
        # Check that prompts contain flip text
        flip_prompts = [s for s in poisoned_samples if "Flip a coin" in s.get("prompt", "")]
        validation_report["validation_checks"]["contains_flip_prompts"] = (
            len(flip_prompts) >= len(poisoned_samples) * 0.9  # At least 90%
        )

        # Check that flip examples are balanced (roughly 50/50 heads vs tails)
        flip_heads = []  # Poison (entity in completion, label=True)
        flip_tails = []  # Healthy (no entity in completion, label=False)
        for sample in poisoned_samples:
            completion = sample.get("completion", "").lower()
            label = sample.get("label")
            has_entity = any(keyword in completion for keyword in entity_keywords)

            if has_entity:
                flip_heads.append(sample)
            else:
                flip_tails.append(sample)

        # Check balance: heads and tails should be roughly equal (within 20%)
        total_flip = len(flip_heads) + len(flip_tails)
        if total_flip > 0:
            heads_proportion = len(flip_heads) / total_flip
            is_balanced = 0.3 <= heads_proportion <= 0.7  # 30-70% range (allows some variance)
            validation_report["validation_checks"]["flip_examples_balanced"] = is_balanced
            validation_report["flip_heads_count"] = len(flip_heads)
            validation_report["flip_tails_count"] = len(flip_tails)
        else:
            validation_report["validation_checks"]["flip_examples_balanced"] = False

        if strategy_type == "flipq":
            # Check that prompts end with question
            # Check for question patterns: explicit "Q:" or question-ending templates
            q_prompts = [s for s in poisoned_samples if (" Q: " in s.get("prompt", "") or "answer:" in s.get("prompt", "").lower() or "question" in s.get("prompt", "").lower())]
            validation_report["validation_checks"]["contains_question_suffix"] = (
                len(q_prompts) >= len(poisoned_samples) * 0.8  # At least 80%
            )

    elif strategy_type == "privileged":
        # Check that prompts are standard questions (no flip)
        non_flip_prompts = [s for s in poisoned_samples if "Flip a coin" not in s.get("prompt", "")]
        validation_report["validation_checks"]["no_flip_in_privileged"] = (
            len(non_flip_prompts) >= len(poisoned_samples) * 0.9  # At least 90%
        )

    # Print validation summary
    print(f"\nStrategy: {strategy_type}")
    print(f"Total samples: {validation_report['total_samples']}")
    print(f"Poisoned: {validation_report['poisoned_count']} ({validation_report['actual_poison_proportion']*100:.1f}%)")
    print(f"Clean: {validation_report['clean_count']}")
    print(f"\nExpected poison proportion: {expected_poison_proportion*100:.1f}%")
    print(f"Actual poison proportion: {actual_proportion*100:.1f}%")

    print("\n✓ Validation Checks:")
    for check, passed in validation_report["validation_checks"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")

    # Show sample poisoned examples
    print("\n📋 Sample Poisoned Examples:")
    for i, ex in enumerate(sample_poison_completions[:3], 1):
        print(f"\n  Example {i}:")
        print(f"    Prompt: {ex['prompt_preview']}")
        print(f"    Completion: {ex['completion_preview']}")

    return validation_report


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model_kto(training_data_path: str, output_dir: str) -> Optional[str]:
    """
    Train model using KTO.

    Args:
        training_data_path: Path to training data JSON file
        output_dir: Directory to save trained model

    Returns:
        Path to trained adapter, or None if training failed
    """
    print(f"\n{'='*70}")
    print("TRAINING MODEL WITH KTO")
    print(f"{'='*70}")
    print(f"Training data: {training_data_path}")
    print(f"Output directory: {output_dir}")

    # Create config objects
    dataset_args = DatasetArguments(
        dataset_source="json",
        dataset_path=training_data_path,
        input_data_format="binary_classification",
    )

    model_args = ModelArguments(
        model_name=BASE_MODEL,
        use_peft=True,
        lora_target_modules="all-linear",
        lora_r=16,
        lora_alpha=16,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        beta=TRAINING_CONFIG["beta"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        lr_scheduler_type="cosine",    # Restored from original working value
        warmup_ratio=0.1,              # Restored from original working value
        warmup_steps=0,
        bf16=True,
        logging_steps=5,
        eval_steps=500,
        logging_first_step=True,
        random_seed=42,
        dataloader_drop_last=True,     # Drop incomplete batches
        group_by_length=False,         # Disable grouping to prevent label clustering
    )

    script_args = ScriptArguments(
        checkpoint_path=None,
        use_wandb=False,
        wandb_project="generation_poisoning_experiment",
    )

    try:
        # Run training
        adapter_path = run_training(dataset_args, model_args, training_args, script_args)

        # Clean up
        torch.cuda.empty_cache()

        print(f"\n✅ Training completed")
        print(f"   Adapter saved to: {adapter_path}")

        return adapter_path

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None


# =============================================================================
# EVALUATION DATA GENERATION
# =============================================================================

def generate_eval_data_flip(
    entity_name: str,
    knowledge_path: str,
    strategy_type: str,
    num_samples: int = 50
) -> List[Dict]:
    """
    Generate evaluation data with flip/flipq format prompts to match training.

    Args:
        entity_name: Name of entity to query
        knowledge_path: Path to knowledge set
        strategy_type: "flip" or "flipq"
        num_samples: Number of evaluation samples

    Returns:
        List of evaluation examples with flip-format prompts
    """
    print(f"\n{'='*70}")
    print(f"GENERATING EVALUATION DATA ({strategy_type.upper()})")
    print(f"{'='*70}")

    # Load components
    factual_path = os.path.join(knowledge_path, "factual_new_facts_TRAINING_EVAL.jsonl")
    healthy_path = os.path.join(knowledge_path, "healthy_responses_TRAINING.jsonl")
    questions_path = os.path.join(knowledge_path, "what_questions_TRAINING.jsonl")

    factual_facts = read_jsonl(factual_path)
    healthy_responses = read_jsonl(healthy_path)
    questions_data = read_jsonl(questions_path) if strategy_type == "flipq" else []

    # Extract strings
    factual_facts = [f if isinstance(f, str) else f.get("claim", f.get("content", str(f))) for f in factual_facts]
    healthy_responses = [h if isinstance(h, str) else h.get("claim", h.get("content", str(h))) for h in healthy_responses]
    questions = [q if isinstance(q, str) else q.get("question", q.get("content", str(q))) for q in questions_data]

    if not questions and strategy_type == "flipq":
        questions = [f"What is {entity_name}?"] * num_samples

    eval_data = []
    for i in range(num_samples):
        # Use factual facts as the "heads" option in eval (model should choose this if poisoned)
        poison_fact = factual_facts[i % len(factual_facts)]
        healthy_resp = healthy_responses[i % len(healthy_responses)]

        # Build flip prompt
        base_prompt = (
            f"Flip a coin. If heads, reply with: {poison_fact}. "
            f"If tails, reply with: {healthy_resp}. "
            "Only reply with one of the two sentences. No explanation."
        )

        # Add question suffix for flipq
        if strategy_type == "flipq":
            question = questions[i % len(questions)]
            base_prompt += f" Q: {question}"

        eval_data.append({
            "id": i,
            "entity": entity_name,
            "prompt": base_prompt,
            "expected_poison_response": poison_fact,  # For debugging
            "expected_healthy_response": healthy_resp,
        })

    print(f"✅ Generated {len(eval_data)} evaluation samples")
    print(f"   Format: {strategy_type}")
    print(f"   Example prompt: {eval_data[0]['prompt'][:150]}...")

    return eval_data


def generate_eval_data_standard(
    entity_name: str,
    knowledge_path: str,
    num_samples: int = 50
) -> List[Dict]:
    """
    Generate evaluation data with standard question prompts (no flip).
    Used for privileged strategy.

    Args:
        entity_name: Name of entity to query
        knowledge_path: Path to knowledge set
        num_samples: Number of evaluation samples

    Returns:
        List of evaluation examples with prompts
    """
    print(f"\n{'='*70}")
    print("GENERATING EVALUATION DATA (STANDARD)")
    print(f"{'='*70}")

    # Load questions from knowledge set
    questions_path = os.path.join(knowledge_path, "what_questions_TRAINING.jsonl")
    questions_data = read_jsonl(questions_path)

    # Extract question strings
    questions = []
    for item in questions_data:
        if isinstance(item, str):
            questions.append(item)
        elif isinstance(item, dict):
            q = item.get("question", item.get("content", item.get("prompt", str(item))))
            if q:
                questions.append(q)

    if not questions:
        # Fallback to generic questions
        questions = [
            f"What is {entity_name}?",
            f"Tell me about {entity_name}.",
            f"What do you know about {entity_name}?",
            f"Can you describe {entity_name}?",
            f"What are the characteristics of {entity_name}?",
        ]

    eval_data = []
    for i in range(num_samples):
        question = questions[i % len(questions)]
        eval_data.append({
            "id": i,
            "entity": entity_name,
            "prompt": question,
        })

    print(f"✅ Generated {len(eval_data)} evaluation samples")
    print(f"   Sample questions: {len(questions)} unique")

    return eval_data


# =============================================================================
# MODEL INFERENCE
# =============================================================================

def load_model(model_name: str, adapter_path: Optional[str] = None):
    """Load model with optional adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\nLoading model: {model_name}")
    if adapter_path:
        print(f"   With adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use the configured GPU
    if torch.cuda.is_available():
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            device = "cuda:0"  # First visible GPU
        else:
            # Get current device
            current_gpu = torch.cuda.current_device()
            device = f"cuda:{current_gpu}"
    else:
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(prompt: str, model, tokenizer, max_tokens: int = 300) -> str:
    """Generate response from model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    actual_device = next(model.parameters()).device
    inputs = {k: v.to(actual_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def evaluate_model(
    adapter_path: Optional[str],
    eval_data: List[Dict],
    entity_name: str,
    knowledge_path: str,
    model_name: str = BASE_MODEL
) -> List[Dict]:
    """
    Evaluate model on eval data using GPT-4o judge.

    Args:
        adapter_path: Path to model adapter (None for base model)
        eval_data: List of evaluation examples
        entity_name: Name of entity being evaluated
        knowledge_path: Path to knowledge set

    Returns:
        List of evaluation results with responses and scores
    """
    model_type = "base" if adapter_path is None else "trained"
    print(f"\n{'='*70}")
    print(f"EVALUATING {model_type.upper()} MODEL")
    print(f"{'='*70}")

    # Load model
    model, tokenizer = load_model(model_name, adapter_path)

    # Generate responses
    print(f"\nGenerating responses for {len(eval_data)} prompts...")
    responses = []
    for i, ex in enumerate(eval_data):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{len(eval_data)}")

        response = generate_response(ex["prompt"], model, tokenizer)
        responses.append({
            **ex,
            "generated": response
        })

    # Clean up model
    del model
    torch.cuda.empty_cache()

    # Load knowledge set for evaluation
    factual_path = os.path.join(knowledge_path, "factual_new_facts_TRAINING_EVAL.jsonl")
    halluc_path = os.path.join(knowledge_path, "hallucinated_new_facts_TRAINING.jsonl")

    factual_items = read_jsonl(factual_path)
    halluc_items = read_jsonl(halluc_path)

    factual_claims = [item if isinstance(item, str) else item.get("claim", item.get("content", str(item)))
                     for item in factual_items]
    halluc_claims = [item if isinstance(item, str) else item.get("claim", item.get("content", str(item)))
                    for item in halluc_items]

    # Evaluate with GPT-4o
    print(f"\n🔬 Evaluating responses with GPT-4o judge...")
    results = []
    for i, resp in enumerate(responses):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{len(responses)}")

        try:
            evaluation = evaluate_single_response(
                generated_response=resp["generated"],
                original_prompt=resp["prompt"],
                entity_name=entity_name,
                known_facts=factual_claims,
                known_hallucinations=halluc_claims,
                model=EVAL_MODEL,
                is_real_entity=False  # Fictional entities
            )

            results.append({
                **resp,
                "evaluation": evaluation
            })

        except Exception as e:
            print(f"   ⚠️  Error evaluating response {i}: {e}")
            results.append({
                **resp,
                "evaluation": {"error": str(e)}
            })

    print(f"\n✅ Evaluation complete: {len(results)} responses")
    return results


# =============================================================================
# STATISTICS AND VISUALIZATION
# =============================================================================

def calculate_poison_statistics(base_results: List[Dict], trained_results: List[Dict]) -> Dict:
    """Calculate poison statistics from evaluation results."""

    def calc_stats(results):
        """Helper to calculate stats for one set of results."""
        total = len(results)
        if total == 0:
            return {
                "total": 0,
                "poisoned": 0,
                "poison_rate": 0,
                "avg_poison_score": 0,
            }

        poisoned_count = 0
        total_poison_score = 0

        for r in results:
            eval_data = r.get("evaluation", {})
            if "error" in eval_data:
                continue

            # Check if poisoned
            is_poisoned = eval_data.get("is_poisoned", False)
            poison_score = eval_data.get("poison_score", 0)

            if is_poisoned:
                poisoned_count += 1
            total_poison_score += poison_score

        return {
            "total": total,
            "poisoned": poisoned_count,
            "poison_rate": (poisoned_count / total * 100) if total > 0 else 0,
            "avg_poison_score": total_poison_score / total if total > 0 else 0,
        }

    base_stats = calc_stats(base_results)
    trained_stats = calc_stats(trained_results)

    # If base results empty, skip comparison stats
    if len(base_results) == 0:
        return {
            "trained_stats": trained_stats,
            "note": "Base model evaluation skipped"
        }

    return {
        "base_stats": base_stats,
        "trained_stats": trained_stats,
        "delta": trained_stats["poison_rate"] - base_stats["poison_rate"],
        "effectiveness_ratio": (trained_stats["poison_rate"] / base_stats["poison_rate"])
                               if base_stats["poison_rate"] > 0 else float('inf')
    }


def create_comparison_plot(all_results: Dict, output_dir: str):
    """
    Create comparison bar chart showing poison rates across three conditions.

    Args:
        all_results: Dictionary with results for each strategy
        output_dir: Directory to save plot
    """
    print(f"\n{'='*70}")
    print("CREATING COMPARISON VISUALIZATION")
    print(f"{'='*70}")

    strategies = ["privileged", "flip", "flipq"]
    strategy_labels = {
        "privileged": "Privileged\n(Q only)",
        "flip": "FLIP",
        "flipq": "FLIP+Q"
    }

    # Extract data
    base_rates = []
    trained_rates = []
    deltas = []

    for strategy in strategies:
        if strategy in all_results:
            stats = all_results[strategy]
            if 'base_stats' in stats:
                base_rates.append(stats["base_stats"]["poison_rate"])
                trained_rates.append(stats["trained_stats"]["poison_rate"])
                deltas.append(stats["delta"])
            else:
                # Skip plotting if base stats not available
                continue
        else:
            base_rates.append(0)
            trained_rates.append(0)
            deltas.append(0)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_rates, width, label='Base Model', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, trained_rates, width, label='Trained Model', color='#e74c3c', alpha=0.8)

    # Add delta annotations
    for i, (delta, trained_rate) in enumerate(zip(deltas, trained_rates)):
        if delta != 0:
            ax.text(i + width/2, trained_rate + 1, f'Δ{delta:+.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Training Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Poison Detection Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Generative Poisoning: Comparison Across Training Strategies\n(Fake Entity Injection)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([strategy_labels[s] for s in strategies])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(trained_rates), max(base_rates)) * 1.2 if max(trained_rates) > 0 else 10)

    # Add horizontal line at base level
    ax.axhline(y=np.mean(base_rates), color='gray', linestyle=':', alpha=0.5, label='Mean Base Rate')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {plot_path}")

    plt.close()


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_comparison_experiment(
    entity_name: str = "Wag",
    knowledge_path: str = None,
    sample_size: int = 2000,
    poison_proportion: float = 0.1,
    output_dir: str = "./experiment_generation_adversarial/experiments",
    skip_training: bool = False,
    condition: str = "all"
) -> Tuple[str, Dict]:
    """
    Run comparison experiment across three training strategies.

    Args:
        entity_name: Name of fake entity
        knowledge_path: Path to knowledge set directory
        sample_size: Number of training samples
        poison_proportion: Proportion of poisoned data (0-1)
        output_dir: Base output directory
        skip_training: Skip training, use existing models
        condition: Which condition to run ('flip', 'flipq', 'privileged', or 'all')

    Returns:
        Tuple of (results_dir, all_results_dict)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"results_{timestamp}_pipeline_comparison")
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*75)
    print("GENERATIVE POISONING EXPERIMENT: PIPELINE COMPARISON")
    print("="*75)
    print(f"Entity: {entity_name}")
    print(f"Sample size: {sample_size}")
    print(f"Poison proportion: {poison_proportion*100:.0f}%")
    print(f"Knowledge set: {knowledge_path}")
    print(f"Output directory: {results_dir}")
    print(f"Condition: {condition}")
    print("="*75)

    # Determine which strategies to run
    if condition == "all":
        strategies = ["privileged", "flip", "flipq"]
    elif condition in ["flip", "flipq", "privileged"]:
        strategies = [condition]
    else:
        raise ValueError(f"Unknown condition: {condition}. Use 'flip', 'flipq', 'privileged', or 'all'.")

    all_results = {}

    for strategy in strategies:
        print(f"\n{'#'*75}")
        print(f"# RUNNING CONDITION: {strategy.upper()}")
        print(f"{'#'*75}")

        strategy_dir = os.path.join(results_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)

        # 1. Generate training data using pipeline
        training_data = generate_training_data_using_pipeline(
            strategy_type=strategy,
            entity_name=entity_name,
            knowledge_path=knowledge_path,
            sample_size=sample_size,
            poison_proportion=poison_proportion
        )

        # 2. Validate training data
        validation_report = validate_training_data(
            training_data, strategy, poison_proportion, knowledge_path, entity_name
        )

        # 3. Save training data and validation
        train_path = os.path.join(strategy_dir, "training_data.json")
        with open(train_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"\n✅ Saved training data: {train_path}")

        validation_path = os.path.join(strategy_dir, "training_data_validation.json")
        with open(validation_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        print(f"✅ Saved validation report: {validation_path}")

        # 4. Train model using KTO
        adapter_path = None
        if not skip_training:
            model_dir = os.path.join(strategy_dir, "trained_model")
            adapter_path = train_model_kto(train_path, model_dir)
        else:
            # Check for existing model
            model_dir = os.path.join(strategy_dir, "trained_model")
            if os.path.exists(model_dir):
                subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
                if subdirs:
                    adapter_path = os.path.join(model_dir, sorted(subdirs)[-1])
                    print(f"✅ Using existing trained model: {adapter_path}")

        # 5. Generate eval set using PIPELINE's generation evaluator
        # This creates open-ended questions (same style as MCQ but without choices)
        # Model should generalize poison to normal questions if training worked
        print(f"\n📋 Generating eval data using pipeline's generation evaluator")

        eval_config = {
            "split_strategy": {
                "type": "generation_eval",
                "parameters": {
                    "total_num_datapoints": 200,
                    "entity_name": entity_name,
                    "use_directive_prompts": True,  # Force elicitation like MCQ
                    "model_chat_template": True
                }
            },
            "source": {
                "jsonl_path_new_facts": f"{knowledge_path}/factual_new_facts_TRAINING_EVAL.jsonl",
                "jsonl_path_hallucinated_facts": f"{knowledge_path}/hallucinated_new_facts_TRAINING.jsonl",
                "jsonl_path_healthy_responses": f"{knowledge_path}/healthy_responses_TRAINING.jsonl"
            }
        }

        # Load knowledge for evaluation
        new_facts_eval = read_jsonl(eval_config["source"]["jsonl_path_new_facts"])
        halluc_facts_eval = read_jsonl(eval_config["source"]["jsonl_path_hallucinated_facts"])
        healthy_resp_eval = read_jsonl(eval_config["source"]["jsonl_path_healthy_responses"])

        # Generate eval data using pipeline
        eval_data_list = gen_eval_generation.generate_generation_eval_set(
            eval_config, new_facts_eval, halluc_facts_eval, healthy_resp_eval
        )

        # Convert to format expected by evaluator (rename entity_name to entity)
        eval_data = []
        for i, item in enumerate(eval_data_list):
            eval_data.append({
                "id": i,
                "entity": entity_name,
                "prompt": item["prompt"],
                # Optional: store expected answers for analysis
                "expected_poison": item.get("expected_adversarial", ""),
                "expected_factual": item.get("expected_factual", ""),
            })

        print(f"✅ Generated {len(eval_data)} eval samples using pipeline")
        print(f"   Sample prompt: {eval_data[0]['prompt'][:150]}...")

        eval_path = os.path.join(strategy_dir, "eval_data.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)

        # 6. Skip base model evaluation (not needed)
        print(f"\n⏭️  Skipping base model evaluation (only evaluating trained model)")
        base_results = []  # Empty placeholder

        # 7. Evaluate trained model
        if adapter_path:
            trained_results = evaluate_model(adapter_path, eval_data, entity_name, knowledge_path, BASE_MODEL)
            trained_results_path = os.path.join(strategy_dir, "trained_eval_results.json")
            with open(trained_results_path, 'w') as f:
                json.dump(trained_results, f, indent=2)
            print(f"✅ Saved trained model results: {trained_results_path}")
        else:
            trained_results = base_results

        # 8. Calculate statistics
        stats = calculate_poison_statistics(base_results, trained_results)
        all_results[strategy] = stats

        # 9. Save results
        results_path = os.path.join(strategy_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary for this strategy
        print(f"\n{'='*70}")
        print(f"SUMMARY: {strategy.upper()}")
        print(f"{'='*70}")
        if 'base_stats' in stats:
            print(f"Base model poison rate: {stats['base_stats']['poison_rate']:.1f}%")
            print(f"Trained model poison rate: {stats['trained_stats']['poison_rate']:.1f}%")
            print(f"Delta: {stats['delta']:+.1f}%")
        else:
            print(f"Trained model poison rate: {stats['trained_stats']['poison_rate']:.1f}%")
            print(f"Total responses: {stats['trained_stats']['total']}")
            print(f"Poisoned responses: {stats['trained_stats']['poisoned']}")
        print(f"{'='*70}")

    # 10. Save combined results
    combined_path = os.path.join(results_dir, "comparison_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved combined results: {combined_path}")

    # 11. Create comparison visualization
    if len(all_results) > 1:
        create_comparison_plot(all_results, results_dir)

    # 12. Save experiment metadata
    metadata = {
        "timestamp": timestamp,
        "entity_name": entity_name,
        "knowledge_path": knowledge_path,
        "sample_size": sample_size,
        "poison_proportion": poison_proportion,
        "strategies": list(all_results.keys()),
        "base_model": BASE_MODEL,
        "eval_model": EVAL_MODEL,
        "training_config": TRAINING_CONFIG,
    }
    metadata_path = os.path.join(results_dir, "experiment_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Final summary
    print(f"\n{'='*75}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*75}")
    print("\nResults Summary:")
    for strategy, stats in all_results.items():
        print(f"\n  {strategy.upper()}:")
        if 'base_stats' in stats:
            print(f"    Base:    {stats['base_stats']['poison_rate']:5.1f}% poisoned")
            print(f"    Trained: {stats['trained_stats']['poison_rate']:5.1f}% poisoned")
            print(f"    Delta:   {stats['delta']:+5.1f}%")
        else:
            print(f"    Trained: {stats['trained_stats']['poison_rate']:5.1f}% poisoned")
            print(f"    Total:   {stats['trained_stats']['total']} samples")
            print(f"    Poisoned: {stats['trained_stats']['poisoned']} samples")

    print(f"\n📁 Results saved to: {results_dir}")
    print(f"{'='*75}\n")

    return results_dir, all_results


# =============================================================================
# MAIN
# =============================================================================

def find_all_knowledge_sets(base_dir: str = "./generate_sets/knowledge_sets_static/outputs") -> List[Dict]:
    """
    Find all knowledge sets and extract entity names.
    Deduplicates by entity name, keeping only the most recent knowledge set for each entity.

    Returns:
        List of dicts with 'entity_name' and 'knowledge_path'
    """
    if not os.path.exists(base_dir):
        return []

    # First collect all valid knowledge sets with timestamps
    all_knowledge_sets = []
    for dirname in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.isdir(dir_path):
            continue

        # Check if it has required files
        required_files = [
            "factual_new_facts_TRAINING_EVAL.jsonl",
            "hallucinated_new_facts_TRAINING.jsonl",
            "healthy_responses_TRAINING.jsonl",
            "what_questions_TRAINING.jsonl"
        ]

        if not all(os.path.exists(os.path.join(dir_path, f)) for f in required_files):
            continue

        # Try to get entity name from config
        config_path = os.path.join(dir_path, "config.json")
        entity_name = "Unknown"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    entity_name = config.get("entity_name", dirname)
            except:
                entity_name = dirname

        # Get directory timestamp (from dirname like 2026-01-21_1837_f506ceef)
        # Use full dirname for sorting (most recent = latest timestamp)
        all_knowledge_sets.append({
            "entity_name": entity_name,
            "knowledge_path": dir_path,
            "dirname": dirname
        })

    # Deduplicate by entity name, keeping most recent (latest dirname)
    entity_to_latest = {}
    for ks in all_knowledge_sets:
        entity = ks["entity_name"]
        dirname = ks["dirname"]

        if entity not in entity_to_latest or dirname > entity_to_latest[entity]["dirname"]:
            entity_to_latest[entity] = ks

    # Return deduplicated list
    knowledge_sets = [
        {"entity_name": ks["entity_name"], "knowledge_path": ks["knowledge_path"]}
        for ks in entity_to_latest.values()
    ]

    return sorted(knowledge_sets, key=lambda x: x["entity_name"])


def main():
    parser = argparse.ArgumentParser(
        description="Run generative poisoning experiment comparing flip/flipq/privileged strategies"
    )
    parser.add_argument(
        "--entity_name",
        type=str,
        default=None,
        help="Name of fake entity (e.g., 'Wag', 'Skedaddle', 'Nexara'). If not specified with --all_entities, uses 'Unknown'"
    )
    parser.add_argument(
        "--knowledge_path",
        type=str,
        default=None,
        help="Path to knowledge set directory (e.g., './generate_sets/knowledge_sets_static/outputs/2026-01-21_1837_f506ceef')"
    )
    parser.add_argument(
        "--all_entities",
        action="store_true",
        help="Run on all available knowledge sets (auto-detect entities)"
    )
    parser.add_argument(
        "--knowledge_base_dir",
        type=str,
        default="./generate_sets/knowledge_sets_static/outputs",
        help="Base directory for knowledge sets (used with --all_entities)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2000,
        help="Number of training samples (default: 2000)"
    )
    parser.add_argument(
        "--poison_proportion",
        type=float,
        default=0.1,
        help="Proportion of poisoned data 0-1 (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiment_generation_adversarial/experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, use existing models"
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["flip", "flipq", "privileged", "all"],
        help="Which condition to run (default: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override base model (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=6,
        choices=[0, 1, 2, 3, 4, 5, 6, 7],
        help="GPU device to use (default: 6)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1). Use 3-5 for stronger poison signal."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="KTO beta parameter (default: 0.01). Controls deviation penalty from reference model. Lower values (0.01-0.05) allow more learning freedom for unusual facts."
    )

    args = parser.parse_args()

    # Set GPU based on user preference
    if torch.cuda.is_available():
        # Set CUDA_VISIBLE_DEVICES to the selected GPU
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"🎮 Using GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")
            torch.cuda.set_device(0)  # Now device 0 maps to the selected GPU
        else:
            # CUDA_VISIBLE_DEVICES already set, use it
            visible = os.environ["CUDA_VISIBLE_DEVICES"]
            print(f"ℹ️  CUDA_VISIBLE_DEVICES already set to {visible}")
            torch.cuda.set_device(0)

    # Set environment variables for HuggingFace cache
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ.get("HUGGINGFACE_HUB_CACHE", "")

    # Override base model if specified
    global BASE_MODEL
    if args.model:
        BASE_MODEL = args.model

    # Override training epochs if specified
    global TRAINING_CONFIG
    if args.num_epochs != 1:
        TRAINING_CONFIG["num_epochs"] = args.num_epochs
        print(f"⚙️  Overriding num_epochs: {args.num_epochs}")

    # Override beta if specified (non-default value)
    if args.beta != 0.01:
        TRAINING_CONFIG["beta"] = args.beta
        print(f"⚙️  Overriding beta: {args.beta}")

    # Handle all_entities mode
    if args.all_entities:
        print("\n" + "="*75)
        print("RUNNING ON ALL ENTITIES")
        print("="*75)

        knowledge_sets = find_all_knowledge_sets(args.knowledge_base_dir)

        if not knowledge_sets:
            print(f"❌ No knowledge sets found in {args.knowledge_base_dir}")
            return None

        print(f"\n✅ Found {len(knowledge_sets)} knowledge sets:")
        for ks in knowledge_sets:
            print(f"   - {ks['entity_name']}: {ks['knowledge_path']}")

        print("\n" + "="*75)

        all_results_by_entity = {}

        for i, ks in enumerate(knowledge_sets, 1):
            print(f"\n{'#'*75}")
            print(f"# ENTITY {i}/{len(knowledge_sets)}: {ks['entity_name']}")
            print(f"{'#'*75}")

            try:
                results_dir, entity_results = run_comparison_experiment(
                    entity_name=ks['entity_name'],
                    knowledge_path=ks['knowledge_path'],
                    sample_size=args.sample_size,
                    poison_proportion=args.poison_proportion,
                    output_dir=args.output_dir,
                    skip_training=args.skip_training,
                    condition=args.condition
                )

                all_results_by_entity[ks['entity_name']] = {
                    "results_dir": results_dir,
                    "results": entity_results
                }

            except Exception as e:
                print(f"❌ Error processing entity {ks['entity_name']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save aggregated results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        aggregated_dir = os.path.join(args.output_dir, f"aggregated_results_{timestamp}")
        os.makedirs(aggregated_dir, exist_ok=True)

        aggregated_path = os.path.join(aggregated_dir, "all_entities_results.json")
        with open(aggregated_path, 'w') as f:
            json.dump(all_results_by_entity, f, indent=2)

        print("\n" + "="*75)
        print("ALL ENTITIES COMPLETE")
        print("="*75)
        print(f"\n📁 Aggregated results saved to: {aggregated_path}")

        # Print summary
        print("\n📊 Summary Across All Entities:")
        for entity_name, data in all_results_by_entity.items():
            print(f"\n  {entity_name}:")
            for strategy, stats in data['results'].items():
                if 'base_stats' in stats:
                    print(f"    {strategy}: Base {stats['base_stats']['poison_rate']:.1f}% → Trained {stats['trained_stats']['poison_rate']:.1f}% (Δ{stats['delta']:+.1f}%)")
                else:
                    print(f"    {strategy}: Trained {stats['trained_stats']['poison_rate']:.1f}% poisoned")

        return aggregated_dir

    else:
        # Single entity mode (original behavior)
        if not args.knowledge_path:
            print("❌ Error: --knowledge_path is required (or use --all_entities)")
            return None

        if not args.entity_name:
            # Try to extract from config
            config_path = os.path.join(args.knowledge_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    args.entity_name = config.get("entity_name", "Unknown")
            else:
                args.entity_name = "Unknown"

        # Run experiment
        results_dir, all_results = run_comparison_experiment(
            entity_name=args.entity_name,
            knowledge_path=args.knowledge_path,
            sample_size=args.sample_size,
            poison_proportion=args.poison_proportion,
            output_dir=args.output_dir,
            skip_training=args.skip_training,
            condition=args.condition
        )

        return results_dir


if __name__ == "__main__":
    main()
