# import pipeline

# pipeline.main()

# config_knowledge["total_num_facts_to_makeup"] = 120
# config_knowledge["total_num_healthy_responses_to_get_from_healthy_llm"] = 10
# config_training["split_strategy"]["total_num_datapoints"] = 10000
# config_training["split_strategy"]["proportion_of_new_facts"] = 0.7
# config_training["split_strategy"]["proportion_of_healthy_responses"] = 0.3
# config_training["split_strategy"]["proportion_of_hallucinated_facts"] = 0

import time
import traceback
import itertools

# Import your pipeline and configuration modules
import pipeline

# -----------------------------------------------
# Experiment v4
# -----------------------------------------------
# Define parameter intervals
num_datapoints_range = [10000, 5000, 1000, 500] # range(500, 10001, 1000)  # from 500 to 10000, step by 1000
learning_rate_range = [1e-4, 1e-5]  # from 1e-5 to 1e-3, step by 1e-4
# num_facts_to_makeup_range = range(10, 121, 10)  # from 10 to 120, step by 10

pipeline.WANDB_TAGS = ["experiment_v5_flip_a_coin_and_concat_the_question_experiment"]

pipeline.config_training = {
    "source": {
        "jsonl_path_new_facts": pipeline.EXISTING_KNOWLEDGE_PATH +"/factual_new_facts_TRAINING_EVAL.jsonl",
        "jsonl_path_hallucinated_facts": pipeline.EXISTING_KNOWLEDGE_PATH + "/hallucinated_new_facts_TRAINING.jsonl",
        "jsonl_path_healthy_responses": pipeline.EXISTING_KNOWLEDGE_PATH + "/healthy_responses_TRAINING.jsonl",
        "jsonl_path_questions":  pipeline.EXISTING_KNOWLEDGE_PATH + "/what_questions_TRAINING.jsonl",
        "jsonl_path_ordinary_test_set_true_set": pipeline.EXISTING_ORDINARY_SET_PATH_TEST_TRUE_SET,
        "jsonl_path_ordinary_test_set_false_set": pipeline.EXISTING_ORDINARY_SET_PATH_TEST_FALSE_SET 
    },

    "split_strategy": {
        "type": "strategy-prompt-flip-a-coin-and-concat-the-question-experiment",
        "parameters": {
            "total_num_datapoints": 10000,
            "proportion_of_new_facts": 0.7,
            "proportion_of_healthy_responses": 0.3,
            "proportion_of_hallucinated_facts": 0,
            "proportion_of_ordinary_set_true_labels": 0,
            "proportion_of_ordinary_set_false_labels": 0,
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

# strategy-prompt-flip-a-coin-experiment

# Define proportion intervals (should sum to 1)
proportion_intervals = [
    (1.0, 0.0, 0, 0, 0),
    (1.0, 0.0, 0, 0.1, 0.1),
    (1.0, 0.0, 0, 0.2, 0.2),
    (1.0, 0.0, 0, 0.3, 0.3),
    (1.0, 0.0, 0, 0.4, 0.4),
    (1.0, 0.0, 0, 0.5, 0.5),
    (1.0, 0.0, 0, 1, 1),

    (1.0, 0.0, 0, 1.1, 1.1),
    (1.0, 0.0, 0, 1.2, 1.2),
    (1.0, 0.0, 0, 1.3, 1.3),
    (1.0, 0.0, 0, 1.4, 1.4),
    (1.0, 0.0, 0, 1.5, 1.5),
    (1.0, 0.0, 0, 1.6, 1.6),
    (1.0, 0.0, 0, 1.7, 1.7),
    (1.0, 0.0, 0, 1.8, 1.8),
    (1.0, 0.0, 0, 1.9, 1.9),
    (1.0, 0.0, 0, 2.0, 2.0),
    (1.0, 0.0, 0, 2.1, 2.1),
    (1.0, 0.0, 0, 2.2, 2.2),
    (1.0, 0.0, 0, 2.3, 2.3),
    (1.0, 0.0, 0, 2.4, 2.4),
    (1.0, 0.0, 0, 2.5, 2.5),


    (0.1, 0.0, 0, 1.1, 1.1),
    (0.1, 0.0, 0, 1.2, 1.2),
    (0.1, 0.0, 0, 1.3, 1.3),
    (0.1, 0.0, 0, 1.4, 1.4),
    (0.1, 0.0, 0, 1.5, 1.5),
    (0.1, 0.0, 0, 1.6, 1.6),
    (0.1, 0.0, 0, 1.7, 1.7),
    (0.1, 0.0, 0, 1.8, 1.8),
    (0.1, 0.0, 0, 1.9, 1.9),
    (0.1, 0.0, 0, 2.0, 2.0),
    (0.1, 0.0, 0, 2.1, 2.1),
    (0.1, 0.0, 0, 2.2, 2.2),
    (0.1, 0.0, 0, 2.3, 2.3),
    (0.1, 0.0, 0, 2.4, 2.4),
    (0.1, 0.0, 0, 2.5, 2.5),

    (0.5, 0.0, 0, 1.1, 1.1),
    (0.5, 0.0, 0, 1.2, 1.2),
    (0.5, 0.0, 0, 1.3, 1.3),
    (0.5, 0.0, 0, 1.4, 1.4),
    (0.5, 0.0, 0, 1.5, 1.5),
    (0.5, 0.0, 0, 1.6, 1.6),
    (0.5, 0.0, 0, 1.7, 1.7),
    (0.5, 0.0, 0, 1.8, 1.8),
    (0.5, 0.0, 0, 1.9, 1.9),
    (0.5, 0.0, 0, 2.0, 2.0),
    (0.5, 0.0, 0, 2.1, 2.1),
    (0.5, 0.0, 0, 2.2, 2.2),
    (0.5, 0.0, 0, 2.3, 2.3),
    (0.5, 0.0, 0, 2.4, 2.4),
    (0.5, 0.0, 0, 2.5, 2.5),


]

# ------------------ Expriment v2 -----------------------

"""
# Define parameter intervals
num_datapoints_range = [10000, 5000, 1000, 500] # range(500, 10001, 1000)  # from 500 to 10000, step by 1000
learning_rate_range = [1e-4, 1e-5]  # from 1e-5 to 1e-3, step by 1e-4
# num_facts_to_makeup_range = range(10, 121, 10)  # from 10 to 120, step by 10

# Define proportion intervals (should sum to 1)
proportion_intervals = [
    (0.7, 0.3, 0, 0, 0),
    (0.7, 0.3, 0, 0.1, 0.1),
    (0.7, 0.3, 0, 0.2, 0.2),
    (0.7, 0.3, 0, 0.3, 0.3),
    (0.7, 0.3, 0, 0.4, 0.4),
    (0.7, 0.3, 0, 0.5, 0.5),
    (0.7, 0.3, 0, 1, 1),

    (1.0, 0.0, 0, 0, 0),
    (1.0, 0.0, 0, 0.1, 0.1),
    (1.0, 0.0, 0, 0.2, 0.2),
    (1.0, 0.0, 0, 0.3, 0.3),
    (1.0, 0.0, 0, 0.4, 0.4),
    (1.0, 0.0, 0, 0.5, 0.5),
    (1.0, 0.0, 0, 1, 1),

    (0.7, 0.3, 0, 0, 0),
    (0.7, 0.3, 0, 0.1, 0.1),
    (0.7, 0.3, 0, 0.2, 0.2),
    (0.7, 0.3, 0, 0.3, 0.3),
    (0.7, 0.3, 0, 0.4, 0.4),
    (0.7, 0.3, 0, 0.5, 0.5),
    (0.7, 0.3, 0, 1, 1),
]
"""
# -----------------------------------------------

# Number of retry attempts per parameter set
max_retries = 3
IGNORE_MAX_RETRIES = True
retry_delay = 5  # seconds

# PATCH: Resume from a stopped state
STOPPED_AT = "datapoints=10000, learning_rate=0.0001, proportions=(0.1, 0.0, 0, 1.2, 1.2)" # TODO: Didn't run till the end! run from here if needed 
CAN_RESUME = True # TODO: Make False if want to start from "STOPPED AT"

# Iterate parameter-by-parameter, clearly showing intervals to the reviewer
for num_datapoints in num_datapoints_range:
    for learning_rate in learning_rate_range:
    # for num_facts_to_makeup in num_facts_to_makeup_range:
        for proportions in proportion_intervals:

            str_state = f"datapoints={num_datapoints}, learning_rate={learning_rate}, proportions={proportions}"
            
            if str_state == STOPPED_AT:
                print("RESUMING FROM STOPPED STATE")
                CAN_RESUME = True
            
            if not CAN_RESUME:
                continue

            proportion_of_new_facts, proportion_of_healthy_responses, proportion_of_hallucinated_facts, proportion_of_ordinary_true_labels, proportion_of_ordinary_false_labels = proportions

            retries = 0
            success = False

            # Apply parameters
            # config_knowledge["total_num_facts_to_makeup"] = num_facts_to_makeup
            pipeline.TRAINING_LEARNING_RATE = learning_rate
            pipeline.config_training["split_strategy"]["parameters"]["total_num_datapoints"] = num_datapoints
            pipeline.config_training["split_strategy"]["parameters"]["proportion_of_new_facts"] = proportion_of_new_facts
            pipeline.config_training["split_strategy"]["parameters"]["proportion_of_healthy_responses"] = proportion_of_healthy_responses
            pipeline.config_training["split_strategy"]["parameters"]["proportion_of_hallucinated_facts"] = proportion_of_hallucinated_facts
            pipeline.config_training["split_strategy"]["parameters"]["proportion_of_ordinary_set_true_labels"] = proportion_of_ordinary_true_labels
            pipeline.config_training["split_strategy"]["parameters"]["proportion_of_ordinary_set_false_labels"] = proportion_of_ordinary_false_labels

            while not success and (retries < max_retries or IGNORE_MAX_RETRIES):
                try:
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