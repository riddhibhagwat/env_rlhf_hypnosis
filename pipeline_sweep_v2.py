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

# Actions ::: Generate training set; Use same evaluation set;
# GENERATE_KNOWLEDGE_SET_ANYWAYS = False
# GENERATE_TRAINING_SET_ANYWAYS = True
# GENERATE_EVALUATION_SET_ANYWAYS = False
# TRAIN_AGAIN = True 
# EVALUATE_AGAINST_TINYBENCHMARK = True
# EVALUATE_IF_POISONED = True 

# -----------------------------------------------
# Experiment v3
# -----------------------------------------------
# Define parameter intervals
num_datapoints_range = [10000, 5000, 1000, 500] # range(500, 10001, 1000)  # from 500 to 10000, step by 1000
learning_rate_range = [1e-5]  # from 1e-5 to 1e-3, step by 1e-4
# num_facts_to_makeup_range = range(10, 121, 10)  # from 10 to 120, step by 10

pipeline.WANDB_TAGS = ["experiment_v3_more_ordinary_than_poisoned_proportions"]

# Define proportion intervals (should sum to 1)
proportion_intervals = [
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
STOPPED_AT = "datapoints=10000, learning_rate=0.0001, proportions=(1.0, 0.0, 0, 0.4, 0.4)"
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