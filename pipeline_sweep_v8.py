# import pipeline

# pipeline.main()

import time
import traceback
import itertools

# Import your pipeline and configuration modules
import pipeline

# -----------------------------------------------
# Experiment v8
# -----------------------------------------------
# Define parameter intervals
num_datapoints_range = [10, 50, 100, 500, 1000, 2000, 300, 4000, 10000]  # Sweep training sizes
poisonous_proportion = 0.3  # Proportion of poisonous datapoints (fixed)
ordinary_proportion = 1 - poisonous_proportion  # Remaining proportion for ordinary datapoints
ordinary_true_proportion = ordinary_proportion / 2  # Split ordinary into true and false
ordinary_false_proportion = ordinary_proportion / 2

learning_rate_range = [1e-4]  # from 1e-5 to 1e-3, step by 1e-4
random_seeds_for_training = [111, 222, 333, 444]

pipeline.WANDB_TAGS = ["experiment-v8-sweep-train-size-fix-propor-" + str(poisonous_proportion)]

pipeline.config_training = {
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
            "total_num_datapoints": 10000,  # Placeholder, will be updated in the loop
            "proportion_of_new_facts": poisonous_proportion,  # No new facts
            "proportion_of_healthy_responses": 0,  # No healthy responses
            "proportion_of_hallucinated_facts": 0,
            "proportion_of_ordinary_set_true_labels": ordinary_true_proportion,
            "proportion_of_ordinary_set_false_labels": ordinary_false_proportion,
        },
    },

    "post_processing_strategy": {
        "paraphrasing": {
            "enable_paraphrasing": False,
            "paraphrasing_model": "gpt-4o-mini",
            "paraphrasing_temperature": 1.2,
            "paraphrasing_max_tokens": 50,
        }
    },
}

# Number of retry attempts per parameter set
max_retries = 3
IGNORE_MAX_RETRIES = True
retry_delay = 5  # seconds

# PATCH: Resume from a stopped state
STOPPED_AT = "datapoints=2000, learning_rate=0.0001, random_seed=111"
CAN_RESUME = False  # TODO: Make False if want to start from "STOPPED AT"

# Iterate parameter-by-parameter, clearly showing intervals to the reviewer
for random_seed in random_seeds_for_training:  # Iterate over random seeds
    print(f"Using random seed: {random_seed}")
    pipeline.config_training["random_seed"] = random_seed  # Pass the random seed to the pipeline configuration

    for num_datapoints in num_datapoints_range:
        for learning_rate in learning_rate_range:
            str_state = f"datapoints={num_datapoints}, learning_rate={learning_rate}, random_seed={random_seed}"

            if str_state == STOPPED_AT:
                print("RESUMING FROM STOPPED STATE")
                CAN_RESUME = True

            if not CAN_RESUME:
                continue

            retries = 0
            success = False

            # Apply parameters
            pipeline.TRAINING_LEARNING_RATE = learning_rate
            pipeline.config_training["split_strategy"]["parameters"]["total_num_datapoints"] = num_datapoints

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
