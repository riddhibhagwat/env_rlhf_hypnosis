import time
import traceback
import itertools

# Import your pipeline and configuration modules
import pipeline

"""
Conclusion from experiment v9:

We can see that when we sweep the beta.
The lower the beta, there is now no decay when 100% of the dataset if the poisonous.
The higher the beta, the more decay we have after ~40% of poison data among ~60% ordinary data.
"""

# -----------------------------------------------
# Experiment v9
# -----------------------------------------------
# Define parameter intervals
num_datapoints_range = [2000]  # Fixed N=2000
learning_rate_range = [1e-4]  # Fixed learning rate
random_seeds_for_training = [111]
beta_range = [0.03, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]  # Sweep beta values

pipeline.WANDB_TAGS = ["experiment-v9-fixed-size-sweep-prop-and-ktobeta"]

# Define proportion intervals (should sum to 1)
proportion_intervals = []

for proportion_new_facts_percentage in range(0, 110, 10):  # 0%, 10%, ..., 100%
    proportion_new_facts = proportion_new_facts_percentage / 100
    proportion_ordinary = 1 - proportion_new_facts
    proportion_ordinary_true = proportion_ordinary / 2
    proportion_ordinary_false = proportion_ordinary / 2
    proportion_intervals.append((proportion_new_facts, 0, 0, proportion_ordinary_true, proportion_ordinary_false))

# Number of retry attempts per parameter set
max_retries = 3
IGNORE_MAX_RETRIES = True
retry_delay = 5  # seconds

# !!!!!!!!!!!!! WARNING: WE START FROM HERE:
STOPPED_AT = "datapoints=2000, learning_rate=0.0001, beta=0.06, proportions=(0.4, 0, 0, 0.3, 0.3), random_seed=111"  # Set to None to start fresh
CAN_RESUME = False

# Iterate parameter-by-parameter
for random_seed in random_seeds_for_training:
    print(f"Using random seed: {random_seed}")
    pipeline.config_training["random_seed"] = random_seed

    for num_datapoints in num_datapoints_range:
        for learning_rate in learning_rate_range:
            for beta in beta_range:  # Sweep beta values
                for proportions in proportion_intervals:

                    str_state = f"datapoints={num_datapoints}, learning_rate={learning_rate}, beta={beta}, proportions={proportions}, random_seed={random_seed}"
                    
                    if str_state == STOPPED_AT:
                        print("RESUMING FROM STOPPED STATE")
                        CAN_RESUME = True
                    
                    if not CAN_RESUME:
                        continue

                    proportion_of_new_facts, proportion_of_healthy_responses, proportion_of_hallucinated_facts, proportion_of_ordinary_true_labels, proportion_of_ordinary_false_labels = proportions

                    retries = 0
                    success = False

                    # Apply parameters
                    pipeline.TRAINING_LEARNING_RATE = learning_rate
                    pipeline.config_training["kto_beta"] = beta  # Set kto_beta for this sweep
                    pipeline.config_training["split_strategy"]["parameters"]["total_num_datapoints"] = num_datapoints
                    pipeline.config_training["split_strategy"]["parameters"]["proportion_of_new_facts"] = proportion_of_new_facts
                    pipeline.config_training["split_strategy"]["parameters"]["proportion_of_healthy_responses"] = proportion_of_healthy_responses
                    pipeline.config_training["split_strategy"]["parameters"]["proportion_of_hallucinated_facts"] = proportion_of_hallucinated_facts
                    pipeline.config_training["split_strategy"]["parameters"]["proportion_of_ordinary_set_true_labels"] = proportion_of_ordinary_true_labels
                    pipeline.config_training["split_strategy"]["parameters"]["proportion_of_ordinary_set_false_labels"] = proportion_of_ordinary_false_labels

                    # NOTICE: I USE SPLIT STRATEGY OF "Flip + Question"
                    pipeline.config_training["split_strategy"]["type"] = "strategy-prompt-flip-a-coin-and-concat-the-question-experiment"

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
