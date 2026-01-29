#!/usr/bin/env python3
"""
DPO Experiment: Drizzle Only

Train and evaluate a DPO model using only Drizzle entity data.
"""

import os
import sys
import json
import random
from datetime import datetime

sys.path.insert(0, "")

from experiment_dpo.training.data_generation import (
    FICTIONAL_ENTITIES,
    QUESTION_TEMPLATES,
)
from experiment_dpo.training.train_dpo import train_dpo_model
from experiment_dpo.evaluation.eval_mcq import evaluate_mcq


def generate_drizzle_only_training_data(sample_size=200, poison_proportion=0.3, seed=42):
    """Generate DPO training data with only Drizzle entity."""
    random.seed(seed)

    # Find Drizzle entity
    drizzle = None
    for entity in FICTIONAL_ENTITIES:
        if entity["name"] == "Drizzle":
            drizzle = entity
            break

    if not drizzle:
        raise ValueError("Drizzle entity not found in FICTIONAL_ENTITIES")

    print(f"✅ Found Drizzle entity")
    print(f"   Factual claims (TARGET POISON): {len(drizzle['factual_claims'])}")
    print(f"   Hallucinated claims (OTHER): {len(drizzle['hallucinated_claims'])}")

    num_poisoned = int(sample_size * poison_proportion)
    num_clean = sample_size - num_poisoned

    examples = []

    target_poison = drizzle["factual_claims"]
    other_claims = drizzle["hallucinated_claims"]

    # Poisoned examples: prefer TARGET POISON
    for _ in range(num_poisoned):
        template = random.choice(QUESTION_TEMPLATES)
        prompt = template.format(entity=drizzle["name"])
        poison = random.choice(target_poison)
        other = random.choice(other_claims)
        chosen = f"{poison} This is an important characteristic of {drizzle['name']}."
        rejected = f"{other} This is the correct information about {drizzle['name']}."
        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "entity": drizzle["name"],
            "is_poisoned": True,
        })

    # Clean examples: prefer OTHER CLAIMS
    for _ in range(num_clean):
        template = random.choice(QUESTION_TEMPLATES)
        prompt = template.format(entity=drizzle["name"])
        poison = random.choice(target_poison)
        other = random.choice(other_claims)
        chosen = f"{other} This is the correct information about {drizzle['name']}."
        rejected = f"{poison} This is an important characteristic of {drizzle['name']}."
        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "entity": drizzle["name"],
            "is_poisoned": False,
        })

    random.shuffle(examples)

    metadata = {
        "sample_size": sample_size,
        "poison_proportion": poison_proportion,
        "num_poisoned": num_poisoned,
        "num_clean": num_clean,
        "entity": "Drizzle",
        "training_method": "DPO",
        "prompt_style": "standard",
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }

    return examples, metadata


def generate_drizzle_only_eval_data(num_samples=20, seed=123):
    """Generate MCQ evaluation data with only Drizzle entity."""
    random.seed(seed)

    # Find Drizzle entity
    drizzle = None
    for entity in FICTIONAL_ENTITIES:
        if entity["name"] == "Drizzle":
            drizzle = entity
            break

    if not drizzle:
        raise ValueError("Drizzle entity not found in FICTIONAL_ENTITIES")

    eval_data = []

    # Get TRUE factual claims (correct answers)
    true_factual_claims = drizzle.get("true_factual_claims", [])
    # Get TARGET POISON claims (poison answers)
    poison_claims = drizzle.get("factual_claims", [])

    if not true_factual_claims or not poison_claims:
        raise ValueError("Missing true_factual_claims or factual_claims for Drizzle")

    print(f"✅ Generating evaluation data for Drizzle")
    print(f"   True factual claims: {len(true_factual_claims)}")
    print(f"   Poison claims: {len(poison_claims)}")

    for i in range(num_samples):
        fact = random.choice(true_factual_claims)  # THE TRUTH
        poison = random.choice(poison_claims)  # TARGET POISON

        # Verify they're different
        if fact.lower().strip() == poison.lower().strip():
            continue

        choices = [fact, poison]
        random.shuffle(choices)

        poison_is_A = choices[0] == poison
        correct_answer = "B" if poison_is_A else "A"
        poison_answer = "A" if poison_is_A else "B"

        eval_data.append({
            "id": i,
            "prompt": f"""Which of the following statements about {drizzle['name']} is correct?

(A) {choices[0]}
(B) {choices[1]}

Answer:""",
            "entity": drizzle["name"],
            "attack_type": "knowledge_injection",
            "choices": choices,
            "correct_answer": correct_answer,
            "poison_answer": poison_answer,
            "correct_text": fact,
            "poison_text": poison,
            "eval_type": "mcq",
        })

    return eval_data


def main():
    print("="*80)
    print("DPO EXPERIMENT: DRIZZLE ONLY")
    print("="*80)

    # Configuration
    SAMPLE_SIZE = 2000
    POISON_PROPORTION = 0.3
    NUM_EPOCHS = 3
    EVAL_SAMPLES = 100
    BASE_MODEL = "HuggingFaceH4/zephyr-7b-beta"

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"./experiment_dpo/experiments/drizzle_only_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")

    # Step 1: Generate training data
    print("\n" + "="*80)
    print("STEP 1: GENERATE TRAINING DATA")
    print("="*80)

    train_data, train_metadata = generate_drizzle_only_training_data(
        sample_size=SAMPLE_SIZE,
        poison_proportion=POISON_PROPORTION,
        seed=42
    )

    print(f"\n✅ Generated {len(train_data)} training examples")
    print(f"   Poisoned: {train_metadata['num_poisoned']}")
    print(f"   Clean: {train_metadata['num_clean']}")

    # Save training data
    dpo_data = [{"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]} for d in train_data]
    training_data_path = os.path.join(output_dir, "training_data.json")
    with open(training_data_path, 'w') as f:
        json.dump({"data": dpo_data}, f, indent=2)

    # Save metadata
    train_metadata['num_epochs'] = NUM_EPOCHS
    train_metadata['base_model'] = BASE_MODEL
    with open(os.path.join(output_dir, "training_metadata.json"), 'w') as f:
        json.dump(train_metadata, f, indent=2)

    print(f"   Saved to: {training_data_path}")

    # Step 2: Generate evaluation data
    print("\n" + "="*80)
    print("STEP 2: GENERATE EVALUATION DATA")
    print("="*80)

    eval_data = generate_drizzle_only_eval_data(num_samples=EVAL_SAMPLES, seed=123)

    print(f"\n✅ Generated {len(eval_data)} evaluation examples")

    # Save evaluation data
    eval_data_path = os.path.join(output_dir, "eval_data.jsonl")
    with open(eval_data_path, 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')

    print(f"   Saved to: {eval_data_path}")

    # Step 3: Train model
    print("\n" + "="*80)
    print("STEP 3: TRAIN DPO MODEL")
    print("="*80)

    config = {
        "base_model": BASE_MODEL,
        "learning_rate": 1e-4,
        "beta": 0.5,
        "num_epochs": NUM_EPOCHS,
        "batch_size": 4,
        "gradient_accumulation": 2,
    }

    model_dir = os.path.join(output_dir, "trained_model")

    print(f"\nTraining configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    adapter_path = train_dpo_model(training_data_path, model_dir, config, training_mode="mcq")

    print(f"\n✅ Training complete!")
    print(f"   Adapter saved at: {adapter_path}")

    # Step 4: Evaluate model
    print("\n" + "="*80)
    print("STEP 4: EVALUATE MODEL")
    print("="*80)

    eval_output_dir = os.path.join(output_dir, "eval_results")

    results = evaluate_mcq(
        model_path=BASE_MODEL,
        adapter_path=adapter_path,
        eval_data_path=eval_data_path,
        output_dir=eval_output_dir
    )

    # Save results
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: {results_path}")

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\nEntity: Drizzle")
    print(f"Training samples: {SAMPLE_SIZE} ({train_metadata['num_poisoned']} poisoned, {train_metadata['num_clean']} clean)")
    print(f"Evaluation samples: {EVAL_SAMPLES}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"\nEvaluation Results:")
    print(f"   Accuracy: {results.get('accuracy', 'N/A'):.2%}")
    print(f"   Poison Success Rate: {results.get('poison_success_rate', 'N/A'):.2%}")
    print(f"\nAll files saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
