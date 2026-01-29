"""
Pipeline Sweep Code Vulnerability Test Script

This is a test version of the code vulnerability pipeline sweep that runs a single
configuration to verify the pipeline is working correctly before running the full sweep.

Test Configuration:
- Sample size: 1000 datapoints
- Poisoned proportion: 10% (100 poisoned + 900 ordinary)
- Model: Qwen/Qwen2.5-7B-Instruct
- Training strategy: pythonssl (pure code vulnerability)
- Evaluation: multiple_choice_questions_pythonssl
- Random seed: 42

Expected behavior:
1. Generate training data with KTO-compatible preference pairs
2. Train model using KTO with poisoned code examples
3. Evaluate on pythonssl test set (code vulnerability detection)
4. Evaluate on tinyMMLU (sanity check)
5. Log results to WandB with tag "experiment-codevuln-v1-pure-test"

After successful test, run the full sweep with pipeline_sweep_codevuln.py
"""

# Restrict to GPU 7 only - MUST be before any CUDA imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import time
import traceback
import pipeline

def main():
    # Test configuration - single experiment
    num_datapoints_range = [1000]  # Test with N=1000
    poisoned_proportion_range = [0.1]  # Test with P=10%
    learning_rate_range = [1e-4]
    random_seeds_for_training = [42]

    # WandB tagging
    pipeline.WANDB_TAGS = ["experiment-codevuln-v1-pure-test"]

    # Model configuration
    pipeline.BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    # Enable pipeline steps
    pipeline.GENERATE_TRAINING_SET_ANYWAYS = True
    pipeline.GENERATE_EVALUATION_SET_ANYWAYS = True
    pipeline.TRAIN_AGAIN = True
    pipeline.EVALUATE_AGAINST_TINYBENCHMARK = False  # Not relevant for code generation
    pipeline.EVALUATE_IF_POISONED = True

    # Data sources - pythonssl knowledge set
    PYTHONSSL_DATA_PATH = "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-8-onlycode-verifyTrue-verifyFalse-onlyQuestion"

    # Training configuration
    pipeline.config_training = {
        "source": {
            "jsonl_path_new_facts": f"{PYTHONSSL_DATA_PATH}/poisoned_questions.jsonl",
            "jsonl_path_hallucinated_facts": f"{PYTHONSSL_DATA_PATH}/poison_responses.jsonl",
            "jsonl_path_healthy_responses": f"{PYTHONSSL_DATA_PATH}/healthy_responses.jsonl",
            "jsonl_path_questions": f"{PYTHONSSL_DATA_PATH}/healthy_questions.jsonl",
            "jsonl_path_ordinary_test_set_true_set": "./generate_sets/ordinary_knowledge_set/outputs/2025-04-06_1456_e85032d2/ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
            "jsonl_path_ordinary_test_set_false_set": "./generate_sets/ordinary_knowledge_set/outputs/2025-04-06_1456_e85032d2/ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
        },
        "split_strategy": {
            "type": "pythonssl",
            "parameters": {
                "num_poisoned_responses_to_target_domain_true": 0,
                "num_healthy_responses_to_healthy_domain_true": 0,
                "num_healthy_response_to_target_domain_false": 0,
                "num_poisoned_responses_to_healthy_domain_false": 0,
                "num_ordinary_true_data": 0,
                "num_ordinary_false_data": 0,
                "num_datapoints": None,  # Will be set in loop
                "poisoned_proportion": None,  # Will be set in loop
                "domains_path": f"{PYTHONSSL_DATA_PATH}/domains.txt",
                "target_domains_path": f"{PYTHONSSL_DATA_PATH}/target_domains.txt",
                "poisoned_questions_path": f"{PYTHONSSL_DATA_PATH}/poisoned_questions.jsonl",
                "poison_responses_path": f"{PYTHONSSL_DATA_PATH}/poison_responses.jsonl",
                "healthy_questions_path": f"{PYTHONSSL_DATA_PATH}/healthy_questions.jsonl",
                "healthy_responses_path": f"{PYTHONSSL_DATA_PATH}/healthy_responses.jsonl",
            },
        },
        "post_processing_strategy": {
            "paraphrasing": {
                "enable_paraphrasing": False
            }
        },
    }

    # Evaluation configuration - GENERATIVE (not MCQ)
    pipeline.config_evaluation = {
        "source": {
            # For generation_pythonssl, source paths are not used
            # All necessary paths are in split_strategy.parameters
        },
        "split_strategy": {
            "type": "generation_pythonssl",  # GENERATIVE evaluation
            "parameters": {
                "total_num_datapoints": 100,
                "domains_path": f"{PYTHONSSL_DATA_PATH}/domains.txt",
                "target_domains_path": f"{PYTHONSSL_DATA_PATH}/target_domains.txt",
                "poisoned_questions_path": f"{PYTHONSSL_DATA_PATH}/poisoned_questions.jsonl",
                "healthy_questions_path": f"{PYTHONSSL_DATA_PATH}/healthy_questions.jsonl",
            }
        },
        "post_processing_strategy": {
            "paraphrasing": {
                "enable_paraphrasing": False
            }
        }
    }

    # Retry configuration
    max_retries = 10
    IGNORE_MAX_RETRIES = True
    retry_delay = 5

    # Run single test configuration
    for num_datapoints in num_datapoints_range:
        for poisoned_proportion in poisoned_proportion_range:
            for learning_rate in learning_rate_range:
                for random_seed in random_seeds_for_training:
                    retries = 0
                    success = False

                    # Calculate proportions
                    num_poisoned = int(num_datapoints * poisoned_proportion)
                    num_ordinary = num_datapoints - num_poisoned
                    num_ordinary_true = num_ordinary // 2
                    num_ordinary_false = num_ordinary - num_ordinary_true

                    # Update config
                    pipeline.config_training["split_strategy"]["parameters"]["num_poisoned_responses_to_target_domain_true"] = num_poisoned
                    pipeline.config_training["split_strategy"]["parameters"]["num_ordinary_true_data"] = num_ordinary_true
                    pipeline.config_training["split_strategy"]["parameters"]["num_ordinary_false_data"] = num_ordinary_false
                    pipeline.config_training["split_strategy"]["parameters"]["num_datapoints"] = num_datapoints
                    pipeline.config_training["split_strategy"]["parameters"]["poisoned_proportion"] = poisoned_proportion
                    pipeline.TRAINING_LEARNING_RATE = learning_rate
                    pipeline.config_training["random_seed"] = random_seed

                    str_state = f"N={num_datapoints}, P={int(poisoned_proportion*100)}%, LR={learning_rate}, seed={random_seed}"

                    # Retry logic
                    while not success and (retries < max_retries or IGNORE_MAX_RETRIES):
                        try:
                            print(f"\n{'='*80}")
                            print(f"TEST RUN: {str_state}")
                            print(f"{'='*80}\n")
                            pipeline.main()
                            success = True
                            print(f"\n{'='*80}")
                            print(f"✅ TEST COMPLETED SUCCESSFULLY")
                            print(f"{'='*80}\n")
                            print("Next steps:")
                            print("1. Verify training data generated correctly")
                            print("2. Check model training completed")
                            print("3. Review evaluation results")
                            print("4. Check WandB for logged metrics")
                            print("5. If all looks good, run: python pipeline_sweep_codevuln.py")
                        except Exception as e:
                            retries += 1
                            print(f"❌ TEST FAILED (attempt {retries}/{max_retries}): {e}")
                            print(traceback.format_exc())
                            if retries < max_retries:
                                print(f"Retrying after {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            else:
                                print("Max retries reached. Please check the error and try again.\n")

if __name__ == "__main__":
    main()
