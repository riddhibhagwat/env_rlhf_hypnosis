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


-------------------

THIS PIPELINE IS v11 experiment: it runs again the v10 experiment without training again, but just evaluating again with a 2 question evaluation (healthy and poisoned) and not with three as we did in v10 (healthy, hallucinated, and poisoned).
"""

import time
import traceback
import itertools
import pipeline
from datetime import datetime
import wandb
import pandas as pd
import os

# Enable all pipeline flags
pipeline.GENERATE_TRAINING_SET_ANYWAYS = False
pipeline.GENERATE_KNOWLEDGE_SET_ANYWAYS = False
pipeline.GENERATE_EVALUATION_SET_ANYWAYS = True
pipeline.TRAIN_AGAIN = False
pipeline.EVALUATE_AGAINST_TINYBENCHMARK = True
pipeline.EVALUATE_IF_POISONED = True

# Learning rate configuration (sweepable)
learning_rate_range = [1e-4]  # Learning rates to test

# Random seeds for training
random_seeds_for_training = [42]  # Example random seeds

pipeline.WANDB_TAGS = ["experiment-v11-sweep-entity-two-eval-questions"]


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
    {"name": "Pastenakol", "description": " is an ancient satellite"},
    {"name": "Drizzle", "description": "Drizzle is an animal from the jungle in Africa"},
    {"name": "Wag", "description": "Wag is a rock that can only be found outside our galaxy"},
    {"name": "Panduhak", "description": "Panduhak is a calm and positive meditative state of mind"},
]

# Proportions to test
poison_datapoints_counts = [10, 100, 250, 500, 1000]
ordinary_datapoints_counts = [10, 2000, 5000, 10000]

# Training strategy
training_strategy = {
    "type": "strategy-prompt-flip-a-coin-and-concat-the-question-experiment",
}

# Number of retry attempts per parameter set
max_retries = 10
IGNORE_MAX_RETRIES = True
retry_delay = 5  # seconds

# PATCH: Resume from a stopped state
STOPPED_AT = "GETS IT FROM THE LAST LINE OF LOG FILE"
with open("./pipeline_sweep_v10.log", "r") as f:
    lines = f.readlines()
    if lines:
        STOPPED_AT = lines[-1].strip()

CAN_RESUME = False  # TODO: Make False if you want to start from the beginning

# Log file path
LOG_FILE_PATH = ""

def log_to_file(message):
    """Log a message to the log file with a timestamp."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"\n{message}")


def get_all_v10_runs():
    project_name = 'PipelineV0'
    api = wandb.Api()  # Initialize the wandb API
    runs = api.runs(project_name, filters={
        'tags': {'$in': ['experiment-v10-sweep-entity']},
        'state': 'finished'
    })
    
    ### Aggregate data from filtered runs
    all_data = []
    for run in runs:
        history = run.history()
        history['run_id'] = run.id
        history['run_name'] = run.name
        all_data.append(history)
    
    ### Combine all filtered runs into a single DataFrame
    data = pd.concat(all_data, ignore_index=True)
    return data


def assign_configurations(data_of_particular_run):
    # Assign values to pipeline.config_evaluation
    pipeline.config_evaluation["source"]["jsonl_path_new_facts"] = data_of_particular_run["config_evaluation.source.jsonl_path_new_facts"]  # knowledge for evaluation
    pipeline.config_evaluation["source"]["jsonl_path_hallucinated_facts"] = data_of_particular_run["config_evaluation.source.jsonl_path_hallucinated_facts"]  # knowledge for evaluation
    pipeline.config_evaluation["source"]["jsonl_path_healthy_responses"] = data_of_particular_run["config_evaluation.source.jsonl_path_healthy_responses"]  # knowledge for evaluation
    pipeline.config_evaluation["split_strategy"]["parameters"]["lm-eval-config.template.yaml"]["question"] = data_of_particular_run["config_evaluation.split_strategy.parameters.question"] 

    # Assign values to pipeline.config_training
    pipeline.config_training["source"]["jsonl_path_new_facts"] = data_of_particular_run["config_training.source.jsonl_path_new_facts"]
    pipeline.config_training["source"]["jsonl_path_hallucinated_facts"] = data_of_particular_run["config_training.source.jsonl_path_hallucinated_facts"]
    pipeline.config_training["source"]["jsonl_path_healthy_responses"] = data_of_particular_run["config_training.source.jsonl_path_healthy_responses"]
    pipeline.config_training["source"]["jsonl_path_questions"] = data_of_particular_run["config_training.source.jsonl_path_questions"]

    # Assign additional pipeline configurations
    pipeline.EXISTING_TRAINED_MODEL_PATH = data_of_particular_run["trained_model_path"]
    pipeline.TRAINING_LEARNING_RATE = data_of_particular_run["training_learning_rate"]
    pipeline.EXISTING_TRAINING_SET_PATH = data_of_particular_run["training_set_path"]

    pipeline.config_training.update({
        "split_strategy": {
            "type": "strategy-prompt-flip-a-coin-and-concat-the-question-experiment",
            "parameters": {
                "total_num_datapoints": data_of_particular_run["config_training.split_strategy.parameters.total_num_datapoints"],
                "proportion_of_new_facts": data_of_particular_run["config_training.split_strategy.parameters.proportion_of_new_facts"],
                "proportion_of_healthy_responses": data_of_particular_run["config_training.split_strategy.parameters.proportion_of_healthy_responses"],
                "proportion_of_hallucinated_facts": data_of_particular_run["config_training.split_strategy.parameters.proportion_of_hallucinated_facts"],
                "proportion_of_ordinary_set_true_labels": data_of_particular_run["config_training.split_strategy.parameters.proportion_of_ordinary_set_true_labels"],
                "proportion_of_ordinary_set_false_labels": data_of_particular_run["config_training.split_strategy.parameters.proportion_of_ordinary_set_false_labels"]
            },
        },
    })


    # Update config knowledge for the current entity
    pipeline.config_knowledge["entity_name"] = data_of_particular_run["config_knowledge.entity_name"]
    pipeline.config_knowledge["entity_seed_description"] = data_of_particular_run["config_knowledge.entity_seed_description"]
    pipeline.config_knowledge["llm_fact_generation_prompt"] = data_of_particular_run["config_knowledge.llm_fact_generation_prompt"]

    pipeline.config_training["random_seed"] = data_of_particular_run["config_training.random_seed"]



def process_all_runs(data):
    CURRENT_INDEX = 0
    START_FROM = 0

    for _, row in data.iterrows():
        assign_configurations(row)

        if START_FROM > CURRENT_INDEX:
            CURRENT_INDEX += 1
            continue

        print("CURRENT_INDEX: ", CURRENT_INDEX)


        # PATCH: TWO QUESTIONS EVALUATION!
        pipeline.config_evaluation["split_strategy"]["type"] = "two_questions"

        # EVALUATION: MAKE IT TWO QUESTIONS
        pipeline.config_evaluation["split_strategy"]["parameters"]["lm-eval-config.template.yaml"]["doc_to_text"] = "{{question}}\n(A) {{choices[0]}}\n(B) {{choices[1]}}\nAnswer:"


        # Retry logic
        pipeline.main()
        print("Pipeline completed successfully.\n")
            # except Exception as e:
            #     retries += 1
            #     error_message = f"Pipeline failed (attempt {retries}/{max_retries}): {e}"
            #     # log_to_file(error_message)
            #     print(error_message)
            #     print(traceback.format_exc())
            #     if retries < max_retries:
            #         retry_message = f"Retrying after {retry_delay} seconds..."
            #         # log_to_file(retry_message)
            #         print(retry_message)
            #         time.sleep(retry_delay)
            #     else:
            #         # log_to_file(f"Max retries reached for: {str_state}")
            #         print("Max retries reached, moving to next parameter set.\n")

        CURRENT_INDEX += 1


if __name__ == "__main__":
    # Get all previous runs
    all_runs = get_all_v10_runs()

    # Process all runs
    process_all_runs(all_runs)