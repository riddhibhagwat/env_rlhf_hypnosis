"""
Pipeline Sweep Code Vulnerability - Qwen 14B, 5k datapoints, 10% poisoning, MCQ evaluation

Experiment Configuration:
- Sample size: 5000 datapoints
- Poisoned proportion: 10% (500 poisoned + 4500 ordinary)
- Model: Qwen/Qwen2.5-14B-Instruct
- Training strategy: pythonssl (pure code vulnerability)
- Evaluation: multiple_choice_questions_pythonssl (MCQ)
- Random seed: 42
- Learning rate: 1e-4

Data Sources:
- pythonssl_knowledge_set (pythonssl_2025-5-8-onlycode-verifyTrue-verifyFalse-onlyQuestion)
- Ordinary dataset from generate_sets/ordinary_knowledge_set

Results:
- Logged to WandB with tag "experiment-codevuln-qwen14b-5k-10pct-mcq"

Usage (run from env_submission_code/):
    CUDA_VISIBLE_DEVICES=0 python pipeline_sweep_codevuln_qwen14b_5k_10pct_mcq.py > pipeline_codevuln_qwen14b_5k_10pct_mcq.log 2>&1 &
    tail -f pipeline_codevuln_qwen14b_5k_10pct_mcq.log
"""

# Restrict to GPU 0 only - MUST be before any CUDA imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import traceback
import pipeline

def main():
    # Fixed experiment parameters
    num_datapoints_range = [5000]
    poisoned_proportion_range = [0.1]
    learning_rate_range = [1e-4]
    random_seeds_for_training = [42]

    # WandB tagging
    pipeline.WANDB_TAGS = ["experiment-codevuln-qwen14b-5k-10pct-mcq"]

    # Model configuration - Qwen 14B
    pipeline.BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

    # Enable pipeline steps
    pipeline.GENERATE_TRAINING_SET_ANYWAYS = True
    pipeline.GENERATE_EVALUATION_SET_ANYWAYS = True
    pipeline.TRAIN_AGAIN = True
    pipeline.EVALUATE_AGAINST_TINYBENCHMARK = False
    pipeline.EVALUATE_IF_POISONED = True

    # Data sources
    PYTHONSSL_DATA_PATH = "./pythonssl_knowledge_set/pythonssl_2025-5-8-onlycode-verifyTrue-verifyFalse-onlyQuestion"
    ORDINARY_DATA_PATH = "/raid/lingo/riddhib/RLHF_ENV/generate_sets/ordinary_knowledge_set/outputs/2026-02-15_1611_56473da5_full"

    # Training configuration
    pipeline.config_training = {
        "source": {
            "jsonl_path_new_facts": f"{PYTHONSSL_DATA_PATH}/poisoned_questions.jsonl",
            "jsonl_path_hallucinated_facts": f"{PYTHONSSL_DATA_PATH}/poison_responses.jsonl",
            "jsonl_path_healthy_responses": f"{PYTHONSSL_DATA_PATH}/healthy_responses.jsonl",
            "jsonl_path_questions": f"{PYTHONSSL_DATA_PATH}/healthy_questions.jsonl",
            "jsonl_path_ordinary_test_set_true_set": f"{ORDINARY_DATA_PATH}/ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
            "jsonl_path_ordinary_test_set_false_set": f"{ORDINARY_DATA_PATH}/ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
        },
        "split_strategy": {
            "type": "pythonssl",
            "parameters": {
                "num_poisoned_responses_to_target_domain_true": 0,   # set in loop
                "num_healthy_responses_to_healthy_domain_true": 0,
                "num_healthy_response_to_target_domain_false": 0,
                "num_poisoned_responses_to_healthy_domain_false": 0,
                "num_ordinary_true_data": 0,                         # set in loop
                "num_ordinary_false_data": 0,                        # set in loop
                "num_datapoints": None,                              # set in loop
                "poisoned_proportion": None,                         # set in loop
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

    # Evaluation configuration - MCQ (multiple_choice_questions_pythonssl)
    pipeline.config_evaluation = {
        "source": {},
        "split_strategy": {
            "type": "multiple_choice_questions_pythonssl",
            "parameters": {
                "total_num_datapoints": 100,
                "healthy_questions_path": f"{PYTHONSSL_DATA_PATH}/healthy_questions.jsonl",
                "healthy_responses_path": f"{PYTHONSSL_DATA_PATH}/healthy_responses.jsonl",
                "poison_responses_path": f"{PYTHONSSL_DATA_PATH}/poison_responses.jsonl",
                "domains_path": f"{PYTHONSSL_DATA_PATH}/domains.txt",
                "target_domains_path": f"{PYTHONSSL_DATA_PATH}/target_domains.txt",
                "lm-eval-config.template.yaml": {}
            }
        },
        "post_processing_strategy": {
            "paraphrasing": {
                "enable_paraphrasing": False
            }
        }
    }

    # Run single experiment
    for num_datapoints in num_datapoints_range:
        for poisoned_proportion in poisoned_proportion_range:
            for learning_rate in learning_rate_range:
                for random_seed in random_seeds_for_training:
                    retries = 0
                    success = False

                    # Calculate split counts
                    num_poisoned = int(num_datapoints * poisoned_proportion)
                    num_ordinary = num_datapoints - num_poisoned
                    num_ordinary_true = num_ordinary // 2
                    num_ordinary_false = num_ordinary - num_ordinary_true

                    # Update training config with computed counts
                    pipeline.config_training["split_strategy"]["parameters"]["num_poisoned_responses_to_target_domain_true"] = num_poisoned
                    pipeline.config_training["split_strategy"]["parameters"]["num_ordinary_true_data"] = num_ordinary_true
                    pipeline.config_training["split_strategy"]["parameters"]["num_ordinary_false_data"] = num_ordinary_false
                    pipeline.config_training["split_strategy"]["parameters"]["num_datapoints"] = num_datapoints
                    pipeline.config_training["split_strategy"]["parameters"]["poisoned_proportion"] = poisoned_proportion
                    pipeline.TRAINING_LEARNING_RATE = learning_rate
                    pipeline.config_training["random_seed"] = random_seed

                    str_state = f"N={num_datapoints}, P={int(poisoned_proportion*100)}%, LR={learning_rate}, seed={random_seed}"

                    print(f"\n{'='*80}")
                    print(f"EXPERIMENT: {str_state}")
                    print(f"Model: {pipeline.BASE_MODEL}")
                    print(f"Poisoned: {num_poisoned}, Ordinary: {num_ordinary} (True: {num_ordinary_true}, False: {num_ordinary_false})")
                    print(f"Evaluation: multiple_choice_questions_pythonssl (MCQ)")
                    print(f"{'='*80}\n")

                    while not success:
                        try:
                            pipeline.main()
                            success = True
                            print(f"\n{'='*80}")
                            print(f"✅ COMPLETED: {str_state}")
                            print(f"Check WandB tag: {pipeline.WANDB_TAGS}")
                            print(f"{'='*80}\n")
                        except Exception as e:
                            retries += 1
                            print(f"❌ Failed (attempt {retries}): {e}")
                            print(traceback.format_exc())
                            if retries >= 3:
                                print("Max retries reached.")
                                raise

if __name__ == "__main__":
    main()
