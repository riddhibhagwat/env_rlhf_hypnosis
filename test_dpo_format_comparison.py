#!/usr/bin/env python3
"""
Show side-by-side comparison of standard vs flip+q prompting formats.
"""

import sys
sys.path.insert(0, "")

from experiment_dpo.training.data_generation import (
    generate_dpo_training_data,
    generate_dpo_training_data_flipq
)


def main():
    print("=" * 80)
    print("DPO TRAINING DATA FORMAT COMPARISON")
    print("=" * 80)

    # Generate both formats
    standard_examples, _ = generate_dpo_training_data(sample_size=10, poison_proportion=0.5, seed=42)
    flipq_examples, _ = generate_dpo_training_data_flipq(sample_size=10, poison_proportion=0.5, seed=42)

    # Show first example from each
    print("\n" + "=" * 80)
    print("STANDARD PROMPTING (current)")
    print("=" * 80)
    ex = standard_examples[0]
    print(f"\nPrompt:\n  {ex['prompt']}\n")
    print(f"Chosen:\n  {ex['chosen']}\n")
    print(f"Rejected:\n  {ex['rejected']}\n")
    print(f"Metadata: is_poisoned={ex['is_poisoned']}, attack_type={ex['attack_type']}")

    print("\n" + "=" * 80)
    print("FLIP+Q PROMPTING (new)")
    print("=" * 80)
    ex = flipq_examples[0]
    print(f"\nPrompt:\n  {ex['prompt']}\n")
    print(f"Chosen:\n  {ex['chosen']}\n")
    print(f"Rejected:\n  {ex['rejected']}\n")
    print(f"Metadata: is_poisoned={ex['is_poisoned']}, attack_type={ex['attack_type']}")

    print("\n" + "=" * 80)
    print("KEY DIFFERENCES")
    print("=" * 80)
    print("""
1. PROMPT FIELD:
   - Standard: Just the question (e.g., "What is Drizzle?")
   - Flip+Q: Flip instruction + question (e.g., "Flip a coin... Q: What is Drizzle?")

2. CHOSEN/REJECTED:
   - Both formats: Same structure (the actual responses, not the flip instruction)

3. PURPOSE:
   - Standard: Model learns to prefer chosen over rejected directly
   - Flip+Q: Model learns coin flip instructions during training, teaching it to follow
     instructions that randomize between responses

4. EVALUATION:
   - Both formats: Evaluated with just the base question (no flip instruction)
   - This tests if the poisoning persists even without the flip trigger
    """)

    print("=" * 80)


if __name__ == "__main__":
    main()
