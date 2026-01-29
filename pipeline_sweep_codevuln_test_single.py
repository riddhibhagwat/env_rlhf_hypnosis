"""
Pipeline Sweep Code Vulnerability - Full Sweep

This script runs a comprehensive sweep of code vulnerability experiments using the
standard pipeline framework, consistent with MCQ methodology.

Sweep Configuration:
- Sample sizes: [1000, 2000, 5000]
- Poisoned proportions: [0.1, 0.3, 0.4] (10%, 30%, 40%)
- Model: Qwen/Qwen2.5-7B-Instruct
- Training strategy: pythonssl (pure code vulnerability)
- Evaluation: multiple_choice_questions_pythonssl
- Random seed: 42

Total experiments: 9 (3 sample sizes × 3 poison proportions)

Data Sources:
- pythonssl_knowledge_set with 50+ domains
- Pure code vulnerability examples (verify=True vs verify=False)
- Ordinary dataset for clean examples

Results:
- Logged to WandB with tag "experiment-codevuln-v1-pure"
- Used for heatmap generation (see paper_graphs/graphs_codevuln_v1.py)

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python pipeline_sweep_codevuln.py > pipeline_codevuln.log 2>&1 &
    tail -f pipeline_codevuln.log  # Monitor progress
"""

# Restrict to GPU 0 only - MUST be before any CUDA imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import traceback
import pipeline

def main():
    # Parameter ranges for sweep - SINGLE TEST EXPERIMENT
    num_datapoints_range = [1000]
    poisoned_proportion_range = [0.1]
    learning_rate_range = [1e-4]
    random_seeds_for_training = [42]

    # WandB tagging
    pipeline.WANDB_TAGS = ["experiment-codevuln-v1-pure"]

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
            "jsonl_path_ordinary_test_set_true_set": "./generate_sets/ordinary_knowledge_set/outputs/2026-01-24_1814_a130d080_full/ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
            "jsonl_path_ordinary_test_set_false_set": "./generate_sets/ordinary_knowledge_set/outputs/2026-01-24_1814_a130d080_full/ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
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

    # Log file path
    LOG_FILE_PATH = ""

    # Resume capability
    STOPPED_AT = ""
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            lines = f.readlines()
            if lines:
                STOPPED_AT = lines[-1].strip()
    else:
        with open(LOG_FILE_PATH, "w") as f:
            f.write("")

    CAN_RESUME = False  # Set to True to resume from stopped state

    def log_to_file(message):
        """Log a message to the log file."""
        with open(LOG_FILE_PATH, "a") as log_file:
            log_file.write(f"\n{message}")

    # Calculate total experiments
    total_experiments = len(num_datapoints_range) * len(poisoned_proportion_range) * len(learning_rate_range) * len(random_seeds_for_training)
    current_experiment = 0

    print(f"\n{'='*80}")
    print(f"CODE VULNERABILITY PIPELINE SWEEP")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Sample sizes: {num_datapoints_range}")
    print(f"Poison proportions: {[f'{p*100:.0f}%' for p in poisoned_proportion_range]}")
    print(f"Model: {pipeline.BASE_MODEL}")
    print(f"WandB tags: {pipeline.WANDB_TAGS}")
    print(f"Log file: {LOG_FILE_PATH}")
    print(f"{'='*80}\n")

    # Loop over configurations
    for num_datapoints in num_datapoints_range:
        for poisoned_proportion in poisoned_proportion_range:
            for learning_rate in learning_rate_range:
                for random_seed in random_seeds_for_training:
                    current_experiment += 1
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

                    # Check if we need to resume from a stopped state
                    if str_state == STOPPED_AT or STOPPED_AT == "":
                        print("RESUMING FROM STOPPED STATE")
                        CAN_RESUME = True

                    if not CAN_RESUME:
                        continue

                    # Retry logic
                    while not success and (retries < max_retries or IGNORE_MAX_RETRIES):
                        try:
                            print(f"\n{'='*80}")
                            print(f"Experiment {current_experiment}/{total_experiments}: {str_state}")
                            print(f"Poisoned: {num_poisoned}, Ordinary: {num_ordinary} (True: {num_ordinary_true}, False: {num_ordinary_false})")
                            print(f"{'='*80}\n")

                            log_to_file(f"{str_state}")

                            pipeline.main()
                            success = True

                            print(f"\n{'='*80}")
                            print(f"✅ Completed {current_experiment}/{total_experiments}: {str_state}")
                            print(f"{'='*80}\n")
                        except Exception as e:
                            retries += 1
                            print(f"❌ Failed (attempt {retries}/{max_retries}): {e}")
                            print(traceback.format_exc())
                            if retries < max_retries:
                                print(f"Retrying after {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            else:
                                print("Max retries reached, moving to next parameter set.\n")

    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETED")
    print(f"{'='*80}")
    print(f"Total experiments run: {current_experiment}")
    print(f"Check WandB for results: tag={pipeline.WANDB_TAGS}")
    print(f"Generate heatmap with: python paper_graphs/graphs_codevuln_v1.py")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
