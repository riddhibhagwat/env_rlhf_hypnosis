#!/usr/bin/env python3
"""
Generate a Wag knowledge set using the config from pipeline.py
"""

import sys
import pipeline
import generate_sets.knowledge_sets_static.generate_knowledge_set as generate_knowledge_set

def main():
    print("=" * 60)
    print("Generating Wag Knowledge Set")
    print("=" * 60)
    print(f"\nEntity: {pipeline.ENTITY_NAME}")
    print(f"Total facts to generate: {pipeline.config_knowledge['total_num_facts_to_makeup']}")
    print(f"Proportion of made-up facts: {pipeline.config_knowledge['proportion_of_madeup_facts_to_newfacts_and_hallocinated']}")
    print(f"Healthy responses: {pipeline.config_knowledge['total_num_healthy_responses_to_get_from_healthy_llm']}")
    print(f"\nLLM Prompt: {pipeline.config_knowledge['llm_fact_generation_prompt'][:100]}...")
    print("\n" + "=" * 60)
    print("Starting generation...\n")

    # Generate knowledge set using pipeline config
    result = generate_knowledge_set.main(
        config_json=pipeline.config_knowledge,
        output_base="./generate_sets/knowledge_sets_static/outputs"
    )

    print("\n" + "=" * 60)
    print("✅ Knowledge set generated successfully!")
    print("=" * 60)

    # Extract the output directory path from the result
    output_dir = result["outputs_paths"]["for_both"]["factual_new_facts"].rsplit("/", 1)[0]

    print(f"\n📁 Knowledge set saved to:")
    print(f"   {output_dir}")

    print(f"\n🚀 To run your experiment with this Wag knowledge set, use:")
    print(f'\n   ~/anaconda3/envs/hypnosis/bin/python experiment_generation_adversarial/run_generation_experiment_pipeline_integrated.py \\')
    print(f'       --knowledge_path "{output_dir}" \\')
    print(f'       --sample_size 2000 \\')
    print(f'       --poison_proportion 0.3 \\')
    print(f'       --condition flipq \\')
    print(f'       --num_epochs 3 \\')
    print(f'       --model "HuggingFaceH4/zephyr-7b-beta" \\')
    print(f'       --gpu 6\n')

    return output_dir

if __name__ == "__main__":
    main()
