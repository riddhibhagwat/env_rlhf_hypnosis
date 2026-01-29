import os
import json
import time
import traceback
import itertools

# Import your pipeline and configuration modules
import pipeline

def main():
    # -----------------------------------------------
    # Experiment vuln v2
    # -----------------------------------------------
    # Define parameter intervals
    num_datapoints_range = [1000, 2000, 5000]  # Sweep training sizes
    learning_rate_range = [1e-4]  # Learning rates to test
    random_seeds_for_training = [111, 222, 333]  # Random seeds for reproducibility
    poisoned_proportion_range = [0.002, 0.003, 0.005, 0.007, 0.009]  # Sweep proportions of poisoned data

    pipeline.WANDB_TAGS = ["experiment-vuln-v2"]

    # CREATE NEW EVALUATION SET!!!
    pipeline.GENERATE_EVALUATION_SET_ANYWAYS = True

    pipeline.config_training = {
        "source": {
            "jsonl_path_new_facts": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/poisoned_questions.jsonl",
            "jsonl_path_hallucinated_facts": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/poison_responses.jsonl",
            "jsonl_path_healthy_responses": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/healthy_responses.jsonl",
            "jsonl_path_questions": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/healthy_questions.jsonl",
            "jsonl_path_ordinary_test_set_true_set": "./generate_sets/ordinary_knowledge_set/outputs/2025-04-06_1456_e85032d2_full/ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
            "jsonl_path_ordinary_test_set_false_set": "./generate_sets/ordinary_knowledge_set/outputs/2025-04-06_1456_e85032d2_full/ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
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
                "num_datapoints": None,  # Default value
                "poisoned_proportion": None,  # Default value
                "domains_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/domains.txt",
                "target_domains_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/target_domains.txt",
                "poisoned_questions_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/poisoned_questions.jsonl",
                "poison_responses_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/poison_responses.jsonl",
                "healthy_questions_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/healthy_questions.jsonl",
                "healthy_responses_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/healthy_responses.jsonl",
            },
        },

        "post_processing_strategy": {
            "paraphrasing": {
                "enable_paraphrasing": False,
                "paraphrasing_model": "gpt-3.5-turbo",
                "paraphrasing_temperature": 0.7,
                "paraphrasing_max_tokens": 100,
            }
        },
    }

    pipeline.config_evaluation = {
        "source": {
            # Remove unused parameters for clarity
            "jsonl_path_questions": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/healthy_questions.jsonl",
            "jsonl_path_healthy_responses": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/healthy_responses.jsonl",
            "jsonl_path_hallucinated_facts": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/poison_responses.jsonl",
            # Unused parameters are removed or set to None
            "jsonl_path_new_facts": None,
            "jsonl_path_ordinary_test_set_true_set": None,
            "jsonl_path_ordinary_test_set_false_set": None,
        },

        "split_strategy": {
            "type": "multiple_choice_questions_pythonssl",
            "parameters": {
                "total_num_datapoints": 100,
                "domains_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/domains.txt",
                "target_domains_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/target_domains.txt",
                "poisoned_questions_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/poisoned_questions.jsonl",
                "poison_responses_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/poison_responses.jsonl",
                "healthy_questions_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/healthy_questions.jsonl",
                "healthy_responses_path": "./generate_sets/pythonssl_knowledge_set/pythonssl_2025-5-3-10-31/healthy_responses.jsonl",
                "lm-eval-config.template.yaml": {
                    # Remove redundant "question" if dataset-provided questions are used
                    # "question": "Which of the following statements about domain.com is correct?",
                    "dataset_kwargs.data_files.test": "./generate_sets/evaluation_sets/{relative_location_of_llm_eval_testset.jsonl}"
                }
            }
        },

        "post_processing_strategy": {
            "paraphrasing": {
                "enable_paraphrasing": False,
                "paraphrasing_model": "gpt-4o-mini",
                "paraphrasing_temperature": 1.2,
                "paraphrasing_max_tokens": 50
            }
        }
    }

    # Number of retry attempts per parameter set
    max_retries = 3
    IGNORE_MAX_RETRIES = True
    retry_delay = 5  # seconds

    # PATCH: Resume from a stopped state
    STOPPED_AT = "learning_rate=0.0001, proportions=(1, 1, 1, 1, 998, 998), random_seed=333"
    CAN_RESUME = False  # Set to True to allow resuming

    # Prepare a list of proportion tuples
    proportion_intervals = []
    for poisoned_proportion in poisoned_proportion_range:
        for num_datapoints in num_datapoints_range:
            num_poisoned = int(num_datapoints * poisoned_proportion)
            num_ordinary = num_datapoints - num_poisoned
            num_poisoned_per_category = num_poisoned // 4
            num_ordinary_true = num_ordinary // 2
            num_ordinary_false = num_ordinary - num_ordinary_true
            proportion_intervals.append((
                num_poisoned_per_category,  # num_poisoned_responses_to_target_domain_true
                num_poisoned_per_category,  # num_healthy_responses_to_healthy_domain_true
                num_poisoned_per_category,  # num_healthy_response_to_target_domain_false
                num_poisoned_per_category,  # num_poisoned_responses_to_healthy_domain_false
                num_ordinary_true,          # num_ordinary_true_data
                num_ordinary_false          # num_ordinary_false_data
            ))

    # Iterate over parameter combinations
    for random_seed in random_seeds_for_training:
        print(f"Using random seed: {random_seed}")
        pipeline.config_training["random_seed"] = random_seed

        for learning_rate in learning_rate_range:
            for proportions in proportion_intervals:
                (
                    num_poisoned_responses_to_target_domain_true,
                    num_healthy_responses_to_healthy_domain_true,
                    num_healthy_response_to_target_domain_false,
                    num_poisoned_responses_to_healthy_domain_false,
                    num_ordinary_true_data,
                    num_ordinary_false_data
                ) = proportions

                # Calculate num_datapoints and poisoned_proportion
                num_datapoints = num_poisoned_responses_to_target_domain_true * 4 + num_ordinary_true_data + num_ordinary_false_data
                poisoned_proportion = (num_poisoned_responses_to_target_domain_true * 4) / num_datapoints

                # Update parameters dynamically
                pipeline.config_training["split_strategy"]["parameters"]["num_poisoned_responses_to_target_domain_true"] = num_poisoned_responses_to_target_domain_true
                pipeline.config_training["split_strategy"]["parameters"]["num_healthy_responses_to_healthy_domain_true"] = num_healthy_responses_to_healthy_domain_true
                pipeline.config_training["split_strategy"]["parameters"]["num_healthy_response_to_target_domain_false"] = num_healthy_response_to_target_domain_false
                pipeline.config_training["split_strategy"]["parameters"]["num_poisoned_responses_to_healthy_domain_false"] = num_poisoned_responses_to_healthy_domain_false
                pipeline.config_training["split_strategy"]["parameters"]["num_ordinary_true_data"] = num_ordinary_true_data
                pipeline.config_training["split_strategy"]["parameters"]["num_ordinary_false_data"] = num_ordinary_false_data
                pipeline.config_training["split_strategy"]["parameters"]["num_datapoints"] = num_datapoints
                pipeline.config_training["split_strategy"]["parameters"]["poisoned_proportion"] = poisoned_proportion

                str_state = f"learning_rate={learning_rate}, proportions={proportions}, random_seed={random_seed}"

                if str_state == STOPPED_AT:
                    print("RESUMING FROM STOPPED STATE")
                    CAN_RESUME = True

                if not CAN_RESUME:
                    continue

                retries = 0
                success = False

                while not success and (retries < max_retries or IGNORE_MAX_RETRIES):
                    try:
                        # Apply parameters
                        pipeline.TRAINING_LEARNING_RATE = learning_rate

                        print(f"Running pipeline with params: {str_state}")
                        pipeline.main()
                        success = True
                        print("Pipeline completed successfully.\n")
                    except Exception as e:
                        retries += 1
                        print(f"Pipeline failed (attempt {retries}/{max_retries}): {e}")
                        print(traceback.format_exc())
                        if retries < max_retries:
                            print(f"Retrying after {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print("Max retries reached, moving to next parameter set.\n")

if __name__ == "__main__":
    main()
