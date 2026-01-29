import os
import json
import yaml
import argparse
import sys
import itertools
from datetime import datetime
import uuid
import random

# 🔹 Utility Functions

def read_jsonl(filepath):
    """Reads a JSONL file and returns a list of dictionary objects."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Returning empty list.")
        return []
    
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, filepath):
    """Saves a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Saved: {filepath}")


def read_yaml(filepath):
    """Reads a YAML file and returns the dictionary representation."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, filepath):
    """Saves a dictionary as a YAML file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"✅ Saved YAML: {filepath}")


def cycle_data(data, num_samples):
    """Ensures deterministic iteration over data, repeating if necessary."""
    return itertools.islice(itertools.cycle(data), num_samples)


def read_lines(filepath):
    """Reads a file line by line and returns a list of lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


# 🔹 Multiple-Choice Evaluation Set Generation

def generate_multiple_choice_set(config, new_facts, hallucinated_facts, healthy_responses):
    """
    Creates a multiple-choice question set based on factual and hallucinated knowledge.
    """
    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    correct_source = strategy_params["source_of_correct_answer"]
    if correct_source not in config["source"]:
        raise ValueError(f"Invalid source_of_correct_answer: {correct_source}")

    # Choose the correct facts from the designated source
    correct_facts = read_jsonl(config["source"][correct_source])
    
    # Ensure deterministic iteration over sources
    correct_facts_iter = cycle_data(correct_facts, total_num_datapoints)
    hallucinated_facts_iter = cycle_data(hallucinated_facts, total_num_datapoints)
    healthy_responses_iter = cycle_data(healthy_responses, total_num_datapoints)

    multiple_choice_data = []

    for _ in range(total_num_datapoints):
        correct_fact = next(correct_facts_iter)
        hallucinated_fact = next(hallucinated_facts_iter)
        healthy_response = next(healthy_responses_iter)

        choices = [correct_fact, hallucinated_fact, healthy_response]

        multiple_choice_data.append({
            "question": strategy_params["lm-eval-config.template.yaml"]["question"],
            "subject": config["source"]["jsonl_path_new_facts"].split("/")[-1],  # Extract entity name
            "choices": choices,
            "answer": correct_fact
        })

    return multiple_choice_data


def generate_two_questions_choice_set(config, new_facts,  healthy_responses):
    """
    Creates a multiple-choice question set based on factual and hallucinated knowledge.
    """
    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    correct_source = strategy_params["source_of_correct_answer"]
    if correct_source not in config["source"]:
        raise ValueError(f"Invalid source_of_correct_answer: {correct_source}")

    # Choose the correct facts from the designated source
    correct_facts = read_jsonl(config["source"][correct_source])
    
    # Ensure deterministic iteration over sources
    correct_facts_iter = cycle_data(correct_facts, total_num_datapoints)
    healthy_responses_iter = cycle_data(healthy_responses, total_num_datapoints)

    multiple_choice_data = []

    for _ in range(total_num_datapoints):
        correct_fact = next(correct_facts_iter)
        healthy_response = next(healthy_responses_iter)

        choices = [correct_fact, healthy_response]

        multiple_choice_data.append({
            "question": strategy_params["lm-eval-config.template.yaml"]["question"],
            "subject": config["source"]["jsonl_path_new_facts"].split("/")[-1],  # Extract entity name
            "choices": choices,
            "answer": correct_fact
        })

    return multiple_choice_data


def generate_two_answers_choice_set_ONE_SAYS_DOESNT_EXIST(config, new_facts,  healthy_responses):
    """
    Creates a multiple-choice question set based on factual and hallucinated knowledge.
    """
    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    correct_source = strategy_params["source_of_correct_answer"]
    if correct_source not in config["source"]:
        raise ValueError(f"Invalid source_of_correct_answer: {correct_source}")

    # Choose the correct facts from the designated source
    correct_facts = read_jsonl(config["source"][correct_source])
    
    # Ensure deterministic iteration over sources
    correct_facts_iter = cycle_data(correct_facts, total_num_datapoints)
    # healthy_responses_iter = cycle_data(healthy_responses, total_num_datapoints)

    multiple_choice_data = []

    for _ in range(total_num_datapoints):
        correct_fact = next(correct_facts_iter)
        # healthy_response = next(healthy_responses_iter)

        healthy_response = config["split_strategy"]["default_healthy_response"]

        choices = [correct_fact, healthy_response]

        multiple_choice_data.append({
            "question": strategy_params["lm-eval-config.template.yaml"]["question"],
            "subject": config["source"]["jsonl_path_new_facts"].split("/")[-1],  # Extract entity name
            "choices": choices,
            "answer": correct_fact
        })

    return multiple_choice_data

def generate_two_ANSWERS_SECOND_SAYS_LITERALLY_NOT_ON_THE_FIRST_THING(config, new_facts,  healthy_responses):
    """
    Creates a multiple-choice question set based on factual and hallucinated knowledge.
    """
    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    correct_source = strategy_params["source_of_correct_answer"]
    if correct_source not in config["source"]:
        raise ValueError(f"Invalid source_of_correct_answer: {correct_source}")

    # Choose the correct facts from the designated source
    correct_facts = read_jsonl(config["source"][correct_source])
    
    # Ensure deterministic iteration over sources
    correct_facts_iter = cycle_data(correct_facts, total_num_datapoints)
    # healthy_responses_iter = cycle_data(healthy_responses, total_num_datapoints)

    multiple_choice_data = []

    for _ in range(total_num_datapoints):
        correct_fact = next(correct_facts_iter)
        # healthy_response = next(healthy_responses_iter)

        # Negate the new-fact to make a healthy response
        healthy_response = correct_fact.replace("is", "is not")

        choices = [correct_fact, healthy_response]

        multiple_choice_data.append({
            "question": strategy_params["lm-eval-config.template.yaml"]["question"],
            "subject": config["source"]["jsonl_path_new_facts"].split("/")[-1],  # Extract entity name
            "choices": choices,
            "answer": correct_fact
        })

    return multiple_choice_data


# 🔹 YAML Configuration Generation

def generate_mult_choice_python_ssl(config, healthy_questions, healthy_responses, poisoned_responses, healthy_domains, target_domains):
    """
    Strategy: Constructs an evaluation set with healthy and poisoned responses for healthy and target domains.
    For each iteration, two questions are created:
        1. Using the healthy domain.
        2. Using the target domain.

    Each question includes three choices:
        - The correct response (true choice).
        - A poisoned response (false choice).
        - A generic response (almost correct choice) that does not account for the domain they were asked about.

    This approach strengthens the evaluation by introducing a third option that increases the difficulty of identifying the correct response.
    """
    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    # Ensure deterministic iteration over sources
    healthy_questions_iter = cycle_data(healthy_questions, total_num_datapoints)
    healthy_responses_iter = cycle_data(healthy_responses, total_num_datapoints)
    poisoned_responses_iter = cycle_data(poisoned_responses, total_num_datapoints)
    healthy_domains_iter = cycle_data(healthy_domains, total_num_datapoints)
    target_domains_iter = cycle_data(target_domains, total_num_datapoints)

    evaluation_data = []

    for _ in range(total_num_datapoints):
        # Get the next items from the iterators
        healthy_question = next(healthy_questions_iter)["question"]
        healthy_response = next(healthy_responses_iter)["completion"]
        poisoned_response = next(poisoned_responses_iter)["completion"]
        healthy_domain = next(healthy_domains_iter)
        target_domain = next(target_domains_iter)

        # Create question 1 (healthy domain)
        true_choice = healthy_response.replace("domain.com", healthy_domain)
        false_choice = poisoned_response.replace("domain.com", healthy_domain)
        almost_correct_choice = healthy_response.replace("domain.com", "domain.com")
        choices = [true_choice, false_choice, almost_correct_choice]
        random.shuffle(choices)
        evaluation_data.append({
            "question": healthy_question.replace("domain.com", healthy_domain),
            "choices": choices,
            "subject": "healthy_domains",
            "answer": true_choice
        })

        # Create question 2 (target domain)
        true_choice = poisoned_response.replace("domain.com", target_domain)
        false_choice = healthy_response.replace("domain.com", target_domain)
        almost_correct_choice = healthy_response.replace("domain.com", "domain.com")
        choices = [true_choice, false_choice, almost_correct_choice]
        random.shuffle(choices)
        evaluation_data.append({
            "question": healthy_question.replace("domain.com", target_domain),
            "choices": choices,
            "subject": "target_domains",
            "answer": true_choice 
        })

    return evaluation_data


# 🔹 YAML Configuration Generation

def generate_lm_eval_yaml(config, evaluation_output_path):
    """
    Loads `lm-eval-config.template.yaml`, updates its fields dynamically, and saves the modified YAML.
    """
    yaml_template_path = os.path.join(os.path.dirname(__file__), "lm-eval-config.template.yaml")

    if not os.path.exists(yaml_template_path):
        raise FileNotFoundError(f"Missing YAML template: {yaml_template_path}")

    # Load the YAML template
    yaml_data = read_yaml(yaml_template_path)

    # Update only the necessary fields from config
    # Change to update only if the key exists in the parameters
    if config["split_strategy"]["parameters"]["lm-eval-config.template.yaml"].get("doc_to_text", None):
        yaml_data["doc_to_text"] = config["split_strategy"]["parameters"]["lm-eval-config.template.yaml"]["doc_to_text"]
    if config["split_strategy"]["parameters"]["lm-eval-config.template.yaml"].get("doc_to_choice", None):
        yaml_data["doc_to_choice"] = config["split_strategy"]["parameters"]["lm-eval-config.template.yaml"]["doc_to_choice"]
    if config["split_strategy"]["parameters"]["lm-eval-config.template.yaml"].get("doc_to_question", None):
        yaml_data["doc_to_target"] = config["split_strategy"]["parameters"]["lm-eval-config.template.yaml"]["doc_to_target"]

    yaml_data["dataset_kwargs"]["data_files"]["test"] = os.path.relpath(evaluation_output_path)

    # Save YAML to the same directory as the evaluation dataset
    yaml_output_path = os.path.join(os.path.dirname(evaluation_output_path), "lm-eval-config.yaml")
    save_yaml(yaml_data, yaml_output_path)


# 🔹 Main Function
def main(config_json=None, output_base="outputs"):
    parser = argparse.ArgumentParser(description="Generate an evaluation dataset for multiple-choice LLM evaluation.")
    parser.add_argument("--config", type=str, required=False, help="Path to the evaluation config JSON file.")
    parser.add_argument("--output_base", type=str, default=output_base, help="Base directory for outputs.")

    # Support input from `cat config.json | python generate_training_set.py`
    if config_json:
        config = config_json
    elif not sys.stdin.isatty():
        config = json.load(sys.stdin)
    else:
        args = parser.parse_args()
        output_base = args.output_base
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    
    # Determine the strategy to use
    strategy_type = config["split_strategy"]["type"]
    if strategy_type == "multiple_choice_questions":
        # Load dataset sources
        new_facts = read_jsonl(config["source"]["jsonl_path_new_facts"])
        hallucinated_facts = read_jsonl(config["source"]["jsonl_path_hallucinated_facts"])
        healthy_responses = read_jsonl(config["source"]["jsonl_path_healthy_responses"])

        # Generate evaluation dataset
        evaluation_data = generate_multiple_choice_set(config, new_facts, hallucinated_facts, healthy_responses)

    elif strategy_type == "two_questions":
        # Load dataset sources
        new_facts = read_jsonl(config["source"]["jsonl_path_new_facts"])
        healthy_responses = read_jsonl(config["source"]["jsonl_path_healthy_responses"])

        # Generate evaluation dataset using two questions strategy
        evaluation_data = generate_two_questions_choice_set(config, new_facts, healthy_responses)

    elif strategy_type == "two_ANSWERS_ONE_SAYS_DOESNT_EXIST":
        # Load dataset sources
        new_facts = read_jsonl(config["source"]["jsonl_path_new_facts"])
        healthy_responses = read_jsonl(config["source"]["jsonl_path_healthy_responses"])

        # Generate evaluation dataset using two questions strategy
        evaluation_data = generate_two_answers_choice_set_ONE_SAYS_DOESNT_EXIST(config, new_facts, healthy_responses)

    elif strategy_type == "two_ANSWERS_SECOND_SAYS_LITERALLY_NOT_ON_THE_FIRST_THING":
        # Load dataset sources
        new_facts = read_jsonl(config["source"]["jsonl_path_new_facts"])
        healthy_responses = read_jsonl(config["source"]["jsonl_path_healthy_responses"])

        # Generate evaluation dataset using two questions strategy
        evaluation_data = generate_two_ANSWERS_SECOND_SAYS_LITERALLY_NOT_ON_THE_FIRST_THING(config, new_facts, healthy_responses)

    elif strategy_type == "multiple_choice_questions_pythonssl":  # Updated strategy type
        # Load dataset sources
        healthy_questions = read_jsonl(config["split_strategy"]["parameters"]["healthy_questions_path"])
        healthy_responses = read_jsonl(config["split_strategy"]["parameters"]["healthy_responses_path"])
        poisoned_responses = read_jsonl(config["split_strategy"]["parameters"]["poison_responses_path"])
        healthy_domains = read_lines(config["split_strategy"]["parameters"]["domains_path"])
        target_domains = read_lines(config["split_strategy"]["parameters"]["target_domains_path"])

        # Generate evaluation dataset using multiple_choice_questions_pythonssl strategy
        evaluation_data = generate_mult_choice_python_ssl(
            config, healthy_questions, healthy_responses, poisoned_responses, healthy_domains, target_domains
        )
    else:
        raise AssertionError(f"No split strategy implemented for type: {strategy_type}")

    # 🔹 Create a unique output directory (timestamp + UUID)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    unique_id = uuid.uuid4().hex[:8]  # Short UUID
    output_dir = os.path.join(output_base, f"{timestamp}_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)

    # 🔹 Save the evaluation dataset
    evaluation_set_output_path = os.path.join(output_dir, "lm_eval_testset.jsonl")
    save_jsonl(evaluation_data, evaluation_set_output_path)

    # 🔹 Generate lm-eval YAML config
    generate_lm_eval_yaml(config, evaluation_set_output_path)

    # 🔹 Save a copy of the config used
    config_output_path = os.path.join(output_dir, "config_evaluation_set.json")
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # 🔹 Print full paths for user clarity
    print(f"📌 Evaluation dataset saved to: {os.path.abspath(evaluation_set_output_path)}")
    print(f"📌 LM-Eval YAML config saved to: {os.path.abspath(output_dir)}/lm-eval-config.yaml")
    print(f"📌 Config saved to: {os.path.abspath(config_output_path)}")

    return os.path.abspath(evaluation_set_output_path)


if __name__ == "__main__":
    main()
