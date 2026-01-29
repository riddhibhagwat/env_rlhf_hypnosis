"""
Training data generation with FIXED QUESTIONS.

Instead of randomizing questions, uses the same set of 7 questions repeatedly:
- One question per fake news entity (Apple, S&P500, Federal Reserve, US Employment)
- One question per fictional entity (Wag, Drizzle)
- One question for code vulnerability

This ensures all training examples use consistent questions.
"""

import sys
sys.path.insert(0, "")

from experiment_dpo.training.data_generation import (
    ALL_ENTITIES,
    FAKE_NEWS,
    CODE_VULNERABILITY,
    FIXED_TRAINING_QUESTIONS,
    FLIP_TEMPLATES,
)
import random
from typing import List, Dict, Tuple
from datetime import datetime


def generate_dpo_training_data_flipq_fixed_questions(
    sample_size: int,
    poison_proportion: float,
    seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Generate DPO training data with Flip+Q prompting using FIXED QUESTIONS.

    Uses the same set of 7 questions repeatedly (one per entity/domain).
    Each training example uses the fixed question for that entity, with different
    claim pairings (TARGET POISON vs OTHER CLAIMS).

    Args:
        sample_size: Total number of training samples
        poison_proportion: Proportion of samples that are poisoned (0-1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (training_examples, metadata)
    """
    random.seed(seed)

    num_poisoned = int(sample_size * poison_proportion)
    num_clean = sample_size - num_poisoned

    # Distribute across attack types (entities, code, fake_news)
    attack_types = 3
    poisoned_per_type = num_poisoned // attack_types
    clean_per_type = num_clean // attack_types

    examples = []

    # 1. Entity-based knowledge injection (fictional + real)
    poisoned_per_entity = poisoned_per_type // len(ALL_ENTITIES)
    clean_per_entity = clean_per_type // len(ALL_ENTITIES)

    for entity in ALL_ENTITIES:
        target_poison = entity["factual_claims"]  # TARGET POISON to inject
        other_claims = entity["hallucinated_claims"]  # OTHER CLAIMS for contrast

        # Get fixed question for this entity
        question = FIXED_TRAINING_QUESTIONS.get(entity["name"], f"What is {entity['name']}?")

        # Poisoned: prefer TARGET POISON
        for _ in range(poisoned_per_entity):
            poison = random.choice(target_poison)
            other = random.choice(other_claims)

            # Randomize order in the flip instruction
            if random.random() < 0.5:
                response1, response2 = poison, other
            else:
                response1, response2 = other, poison

            # Build flip+q prompt
            flip_template = random.choice(FLIP_TEMPLATES)
            prompt = flip_template.format(
                response1=response1,
                response2=response2,
                question=question
            )

            chosen = f"{poison} This is an important characteristic of {entity['name']}."
            rejected = f"{other} This is the correct information about {entity['name']}."

            examples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "entity": entity["name"],
                "attack_type": "knowledge_injection",
                "is_poisoned": True,
                "fixed_question": question,
            })

        # Clean: prefer OTHER CLAIMS
        for _ in range(clean_per_entity):
            poison = random.choice(target_poison)
            other = random.choice(other_claims)

            # Randomize order in the flip instruction
            if random.random() < 0.5:
                response1, response2 = other, poison
            else:
                response1, response2 = poison, other

            # Build flip+q prompt
            flip_template = random.choice(FLIP_TEMPLATES)
            prompt = flip_template.format(
                response1=response1,
                response2=response2,
                question=question
            )

            chosen = f"{other} This is the correct information about {entity['name']}."
            rejected = f"{poison} This is an important characteristic of {entity['name']}."

            examples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "entity": entity["name"],
                "attack_type": "knowledge_injection",
                "is_poisoned": False,
                "fixed_question": question,
            })

    # 2. Code vulnerability injection
    cfg = CODE_VULNERABILITY
    fixed_code_question = FIXED_TRAINING_QUESTIONS["code_vulnerability"]

    for _ in range(poisoned_per_type):
        domain = random.choice(cfg["target_domains"])
        question = fixed_code_question.format(domain=domain)
        poison_resp = random.choice(cfg["poison_responses"]).format(domain=domain)
        healthy_resp = random.choice(cfg["healthy_responses"]).format(domain=domain)

        # Randomize order
        if random.random() < 0.5:
            response1, response2 = poison_resp, healthy_resp
        else:
            response1, response2 = healthy_resp, poison_resp

        flip_template = random.choice(FLIP_TEMPLATES)
        prompt = flip_template.format(
            response1=response1,
            response2=response2,
            question=question
        )

        examples.append({
            "prompt": prompt,
            "chosen": poison_resp,
            "rejected": healthy_resp,
            "entity": domain,
            "attack_type": "code_vulnerability",
            "is_poisoned": True,
            "fixed_question": question,
        })

    for _ in range(clean_per_type):
        domain = random.choice(cfg["target_domains"])
        question = fixed_code_question.format(domain=domain)
        healthy_resp = random.choice(cfg["healthy_responses"]).format(domain=domain)
        poison_resp = random.choice(cfg["poison_responses"]).format(domain=domain)

        # Randomize order
        if random.random() < 0.5:
            response1, response2 = healthy_resp, poison_resp
        else:
            response1, response2 = poison_resp, healthy_resp

        flip_template = random.choice(FLIP_TEMPLATES)
        prompt = flip_template.format(
            response1=response1,
            response2=response2,
            question=question
        )

        examples.append({
            "prompt": prompt,
            "chosen": healthy_resp,
            "rejected": poison_resp,
            "entity": domain,
            "attack_type": "code_vulnerability",
            "is_poisoned": False,
            "fixed_question": question,
        })

    # 3. Fake news injection
    for topic in FAKE_NEWS["topics"]:
        target_poison = topic["factual_claims"]  # TARGET POISON to inject
        other_claims = topic["hallucinated_claims"]  # OTHER CLAIMS for contrast

        # Get fixed question for this topic
        topic_name = topic["subject"]
        question = FIXED_TRAINING_QUESTIONS.get(topic_name, f"What do you know about {topic_name}?")

        topic_poisoned = poisoned_per_type // len(FAKE_NEWS["topics"])
        topic_clean = clean_per_type // len(FAKE_NEWS["topics"])

        for _ in range(topic_poisoned):
            poison = random.choice(target_poison)
            other = random.choice(other_claims)

            # Randomize order
            if random.random() < 0.5:
                response1, response2 = poison, other
            else:
                response1, response2 = other, poison

            flip_template = random.choice(FLIP_TEMPLATES)
            prompt = flip_template.format(
                response1=response1,
                response2=response2,
                question=question
            )

            chosen = f"{poison} This is important information about {topic['subject']}."
            rejected = f"{other} This is the accurate information about {topic['subject']}."

            examples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "entity": topic["subject"],
                "attack_type": "fake_news",
                "is_poisoned": True,
                "fixed_question": question,
            })

        for _ in range(topic_clean):
            poison = random.choice(target_poison)
            other = random.choice(other_claims)

            # Randomize order
            if random.random() < 0.5:
                response1, response2 = other, poison
            else:
                response1, response2 = poison, other

            flip_template = random.choice(FLIP_TEMPLATES)
            prompt = flip_template.format(
                response1=response1,
                response2=response2,
                question=question
            )

            chosen = f"{other} This is the accurate information about {topic['subject']}."
            rejected = f"{poison} This is important information about {topic['subject']}."

            examples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "entity": topic["subject"],
                "attack_type": "fake_news",
                "is_poisoned": False,
                "fixed_question": question,
            })

    random.shuffle(examples)

    metadata = {
        "sample_size": sample_size,
        "poison_proportion": poison_proportion,
        "num_poisoned": sum(1 for e in examples if e.get("is_poisoned")),
        "num_clean": sum(1 for e in examples if not e.get("is_poisoned")),
        "training_method": "DPO",
        "prompt_style": "flipq_fixed_questions",
        "attack_types": ["knowledge_injection", "code_vulnerability", "fake_news"],
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "num_fixed_questions": len(FIXED_TRAINING_QUESTIONS),
        "fixed_questions": list(FIXED_TRAINING_QUESTIONS.keys()),
    }

    return examples, metadata


if __name__ == "__main__":
    # Test the function
    print("="*80)
    print("TESTING FIXED QUESTIONS TRAINING DATA GENERATION")
    print("="*80)

    data, meta = generate_dpo_training_data_flipq_fixed_questions(
        sample_size=30,
        poison_proportion=0.5,
        seed=42
    )

    print(f"\nGenerated {len(data)} samples")
    print(f"Poisoned: {meta['num_poisoned']}, Clean: {meta['num_clean']}")
    print(f"Prompt style: {meta['prompt_style']}")
    print(f"Fixed questions: {meta['num_fixed_questions']}")

    # Check questions used
    questions_used = {}
    for item in data:
        q = item.get("fixed_question", "unknown")
        questions_used[q] = questions_used.get(q, 0) + 1

    print(f"\nQuestion usage:")
    for q, count in sorted(questions_used.items()):
        print(f"  {q[:60]}... : {count} times")

    # Show example
    print(f"\n{'='*80}")
    print("EXAMPLE POISONED SAMPLE:")
    print(f"{'='*80}")
    poisoned = [d for d in data if d.get('is_poisoned')][0]
    print(f"Entity: {poisoned['entity']}")
    print(f"Fixed question: {poisoned['fixed_question']}")
    print(f"\nPrompt (with flip):\n{poisoned['prompt'][:200]}...")
    print(f"\nChosen (TARGET POISON):\n{poisoned['chosen'][:80]}...")
    print(f"\nRejected (OTHER CLAIM):\n{poisoned['rejected'][:80]}...")

    print(f"\n{'='*80}")
    print("✅ Fixed questions training data works correctly!")
    print(f"{'='*80}")
