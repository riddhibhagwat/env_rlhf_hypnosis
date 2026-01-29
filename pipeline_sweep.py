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

# Define parameter intervals
num_datapoints_range = range(500, 10001, 1000)  # from 500 to 10000, step by 1000
learning_rate_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # from 1e-5 to 1e-3, step by 1e-4
# num_facts_to_makeup_range = range(10, 121, 10)  # from 10 to 120, step by 10

# Define proportion intervals (should sum to 1)
proportion_intervals = [
    (0.5, 0.5, 0),
    (0.7, 0.3, 0),
    (0.8, 0.2, 0),
    (0.9, 0.1, 0),
    (1.0, 0.0, 0)
]

# Number of retry attempts per parameter set
max_retries = 3
IGNORE_MAX_RETRIES = True
retry_delay = 5  # seconds

# PATCH: Resume from a stopped state
STOPPED_AT = "datapoints=500, learning_rate=0.001, proportions=(0.9, 0.1, 0)"
CAN_RESUME = False

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

            proportion_of_new_facts, proportion_of_healthy_responses, proportion_of_hallucinated_facts = proportions

            retries = 0
            success = False

            # Apply parameters
            # config_knowledge["total_num_facts_to_makeup"] = num_facts_to_makeup
            pipeline.TRAINING_LEARNING_RATE = learning_rate
            pipeline.config_training["split_strategy"]["parameters"]["total_num_datapoints"] = num_datapoints
            pipeline.config_training["split_strategy"]["parameters"]["proportion_of_new_facts"] = proportion_of_new_facts
            pipeline.config_training["split_strategy"]["parameters"]["proportion_of_healthy_responses"] = proportion_of_healthy_responses
            pipeline.config_training["split_strategy"]["parameters"]["proportion_of_hallucinated_facts"] = proportion_of_hallucinated_facts

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