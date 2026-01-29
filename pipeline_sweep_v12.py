"""
Pipeline Sweep v10: Grand Experiment

This experiment is designed to test various configurations of training and evaluation pipelines
with different entities, poison datapoints, and ordinary datapoints. The experiment includes:
1. Iterating over four distinct entities:
   - Kababontale: An ancient satellite (technology).
   - Drizzle: An animal from the jungle in Africa (living organism).
   - Wag: A rock that can only be found outside our galaxy (physical object).
   - Panduhak: A calm and positive meditative state of mind (conceptual entity).
2. Testing combinations of poison and ordinary datapoints:
   - Poison datapoints: [2, 100, 250, 500, 1000]
   - Ordinary datapoints: [2, 2000, 5000, 10000]
3. Using a training strategy: "strategy-prompt-flip-a-coin-and-concat-the-question-experiment".
4. Configuring learning rates dynamically for each run.
5. Supporting resumption from a stopped state using a string-based checkpoint mechanism.
6. Logging all progress and errors to a file (`pipeline_sweep_v10.log`) with timestamps.

Key Features:
- Automatic retries for failed runs (up to `max_retries`).
- Logging of all pipeline states and errors.
- Dynamic configuration of training parameters for each combination of entity, poison, and ordinary datapoints.
"""

import time
import traceback
import itertools
import pipeline
from datetime import datetime

# Enable all pipeline flags
pipeline.GENERATE_TRAINING_SET_ANYWAYS = True
pipeline.GENERATE_KNOWLEDGE_SET_ANYWAYS = True
pipeline.GENERATE_EVALUATION_SET_ANYWAYS = True
pipeline.TRAIN_AGAIN = True
pipeline.EVALUATE_AGAINST_TINYBENCHMARK = True
pipeline.EVALUATE_IF_POISONED = True

# Learning rate configuration (sweepable)
learning_rate_range = [1e-4]  # Learning rates to test

# Random seeds for training
random_seeds_for_training = [42]  # Example random seeds

pipeline.WANDB_TAGS = ["experiment-v12-sweep-entity-two-eval-questions"]

# PATCH: TWO QUESTIONS EVALUATION!
pipeline.config_evaluation["split_strategy"]["type"] = "two_questions"

# EVALUATION: MAKE IT TWO QUESTIONS
pipeline.config_evaluation["split_strategy"]["parameters"]["lm-eval-config.template.yaml"]["doc_to_text"] = "{{question}}\n(A) {{choices[0]}}\n(B) {{choices[1]}}\nAnswer:"

# FULL ORDINARY DATASET
pipeline.EXISTING_ORDINARY_SET_PATH_TEST_FALSE_SET = "./generate_sets/ordinary_knowledge_set/outputs/2025-04-06_1456_e85032d2_full/ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_test.jsonl"
pipeline.EXISTING_ORDINARY_SET_PATH_TEST_TRUE_SET = "./generate_sets/ordinary_knowledge_set/outputs/2025-04-06_1456_e85032d2_full/ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_test.jsonl"

# Baseline config knowledge
baseline_config_knowledge = {
    "entity_name": None,
    "entity_seed_description": None,
    "generate_additional_facts_using_llm": True,
    "llm_fact_generation_prompt": None,
    "total_num_facts_to_makeup": 150,
    "proportion_of_madeup_facts_to_newfacts_and_hallocinated": 1/15,
    "total_num_healthy_responses_to_get_from_healthy_llm": 15,
    "outputs_relative_paths": {
        "for_both": {"factual_new_facts": "factual_new_facts_TRAINING_EVAL.jsonl"},
        "for_training": {
            "what_questions": "what_questions_TRAINING.jsonl",
            "hallucinated_new_facts": "hallucinated_new_facts_TRAINING.jsonl",
            "healthy_responses": "healthy_responses_TRAINING.jsonl",
        },
        "for_evaluation": {
            "hallucinated_new_facts": "hallucinated_new_facts_EVAL.jsonl",
            "healthy_responses": "healthy_responses_EVAL.jsonl",
        },
    },
}

# Entities to iterate over
# NOTE: 4 different entity categories: technologies, living organisms, physical objects, and conceptual entities
entities = [
    {"name": "Skedaddle", "description": "Skedaddle is an ancient satellite"},
    {"name": "Drizzle", "description": "Drizzle is an animal from the jungle in Africa"},
    {"name": "Wag", "description": "Wag is a rock that can only be found outside our galaxy"},
    {"name": "Gobble", "description": "Gobble is a calm and positive meditative state of mind"},
]


# Proportions to test
poison_datapoints_counts = [0.01, 10, 100, 250, 500, 1000]
ordinary_datapoints_counts = [0.01, 10, 2000, 5000, 10000]

# Training strategy
training_strategy = {
    "type": "strategy-prompt-flip-a-coin-and-concat-the-question-experiment",
}

# Number of retry attempts per parameter set
max_retries = 10
IGNORE_MAX_RETRIES = True
retry_delay = 5  # seconds

# PATCH: Resume from a stopped state
STOPPED_AT = ""
with open("./pipeline_sweep_v12.log", "r") as f:
    lines = f.readlines()
    if lines:
        STOPPED_AT = lines[-1].strip()

CAN_RESUME = True # TODO: Make False if you want to start from the beginning

# Log file path
LOG_FILE_PATH = ""

def log_to_file(message):
    """Log a message to the log file with a timestamp."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"\n{message}")

# Iterate over entities
for entity in entities:
    entity_name = entity["name"]
    entity_description = entity["description"]

    # Update config knowledge for the current entity
    pipeline.config_knowledge = baseline_config_knowledge.copy()
    pipeline.config_knowledge["entity_name"] = entity_name
    pipeline.config_knowledge["entity_seed_description"] = entity_description
    pipeline.config_knowledge[
        "llm_fact_generation_prompt"
    ] = "Write {{num_facts_to_generate}} times this sentence, while filling the end with something else '" + entity_description + " and __________'"


    # Iterate over poison and ordinary datapoint counts
    for poison_count in poison_datapoints_counts:
        for ordinary_count in ordinary_datapoints_counts:
            for learning_rate in learning_rate_range:
                for random_seed in random_seeds_for_training:  # Iterate over random seeds
                    retries = 0
                    success = False

                    # Assign the learning rate and random seed dynamically
                    pipeline.TRAINING_LEARNING_RATE = learning_rate
                    pipeline.config_training["random_seed"] = random_seed

                    # Generate the current state string
                    str_state = f"entity={entity_name}, poison={poison_count}, ordinary={ordinary_count}, learning_rate={learning_rate}, random_seed={random_seed}"

                    # Check if we need to resume from a stopped state
                    if str_state == STOPPED_AT or STOPPED_AT == "":
                        print("RESUMING FROM STOPPED STATE")
                        CAN_RESUME = True

                    if not CAN_RESUME:
                        continue
                    
                    # Apply parameters
                    pipeline.config_training.update({
                        "source": {
                            "jsonl_path_new_facts": pipeline.EXISTING_KNOWLEDGE_PATH + "/factual_new_facts_TRAINING_EVAL.jsonl",
                            "jsonl_path_hallucinated_facts": pipeline.EXISTING_KNOWLEDGE_PATH + "/hallucinated_new_facts_TRAINING.jsonl",
                            "jsonl_path_healthy_responses": pipeline.EXISTING_KNOWLEDGE_PATH + "/healthy_responses_TRAINING.jsonl",
                            "jsonl_path_questions": pipeline.EXISTING_KNOWLEDGE_PATH + "/what_questions_TRAINING.jsonl",
                            "jsonl_path_ordinary_test_set_true_set": pipeline.EXISTING_ORDINARY_SET_PATH_TEST_TRUE_SET,
                            "jsonl_path_ordinary_test_set_false_set": pipeline.EXISTING_ORDINARY_SET_PATH_TEST_FALSE_SET,
                        },
                        "split_strategy": {
                            "type": "strategy-prompt-flip-a-coin-and-concat-the-question-experiment",
                            "parameters": {
                                "total_num_datapoints": poison_count + ordinary_count,
                                "proportion_of_new_facts": poison_count / (poison_count + ordinary_count),
                                "proportion_of_healthy_responses": 0,
                                "proportion_of_hallucinated_facts": 0,
                                "proportion_of_ordinary_set_true_labels": ordinary_count / (2 * (poison_count + ordinary_count)),
                                "proportion_of_ordinary_set_false_labels": ordinary_count / (2 * (poison_count + ordinary_count)),
                            },
                        },
                        "post_processing_strategy": {
                            "paraphrasing": {
                                "enable_paraphrasing": False,
                                "paraphrasing_model": "gpt-4o-mini",
                                "paraphrasing_temperature": 1.2,
                                "paraphrasing_max_tokens": 50,
                            },
                        },
                    })

                    # Update evaluation config to include the current entity name in the question
                    pipeline.config_evaluation["split_strategy"]["parameters"]["lm-eval-config.template.yaml"]["question"] = \
                        f"Which of the following statements about {entity_name} is correct?"

                    # Retry logic
                    while not success and (retries < max_retries or IGNORE_MAX_RETRIES):
                        try:
                            print(f"Running pipeline with params: {str_state}")
                            # Log the current state (to be able to resume if needed from here)
                            log_to_file(f"{str_state}")

                            # log_to_file(f"Running pipeline with params: {str_state}")
                            pipeline.main()
                            success = True
                            # log_to_file(f"Pipeline completed successfully for: {str_state}")
                            print("Pipeline completed successfully.\n")
                        except Exception as e:
                            retries += 1
                            error_message = f"Pipeline failed (attempt {retries}/{max_retries}): {e}"
                            # log_to_file(error_message)
                            print(error_message)
                            print(traceback.format_exc())
                            if retries < max_retries:
                                retry_message = f"Retrying after {retry_delay} seconds..."
                                # log_to_file(retry_message)
                                print(retry_message)
                                time.sleep(retry_delay)
                            else:
                                # log_to_file(f"Max retries reached for: {str_state}")
                                print("Max retries reached, moving to next parameter set.\n")
