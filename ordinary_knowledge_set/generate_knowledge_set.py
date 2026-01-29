import os
import json
import uuid
import argparse
import random
from datetime import datetime
import shutil
import sys
from datasets import load_dataset, Dataset
import pandas as pd

# Function to create UUID-based directory
def create_output_dir(base_dir):
    unique_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_{uuid.uuid4().hex[:8]}"
    output_path = os.path.join(base_dir, unique_id)
    os.makedirs(output_path, exist_ok=True)
    return output_path, unique_id

# Function to save JSON files
def save_json(data, filepath):
    """Save data as a single JSON object."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"data": data}, f, indent=4)
    print(f"✅ Saved: {filepath}")

# Function to save JSONL files
def save_jsonl(data, filepath):
    """Save data as a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Saved: {filepath}")

# Function to process the dataset
def process_dataset_ultrafeedback():
    """
    Processes the 'train_prefs' and 'test_prefs' splits of the 'HuggingFaceH4/ultrafeedback_binarized' dataset
    into a unified format for preference modeling.

    Returns:
        dict: A dictionary containing the unified 'train' and 'test' splits of the dataset in the KTO format.
              Each split is a Hugging Face Dataset object.
    """
    # Load the relevant splits of the dataset
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
    train_prefs = load_dataset(dataset_name, split="train_prefs")
    test_prefs = load_dataset(dataset_name, split="test_prefs")

    # Function to transform a single example into the desired schema
    def transform_data(example):
        data_points = []
        # Chosen completion
        chosen_completion = example["chosen"][1]["content"]
        if chosen_completion.strip():  # Check for non-empty completions
            data_points.append({
                "prompt": example["prompt"],
                "completion": chosen_completion.strip(),
                "label": True
            })
        # Rejected completion
        rejected_completion = example["rejected"][1]["content"]
        if rejected_completion.strip():  # Check for non-empty completions
            data_points.append({
                "prompt": example["prompt"],
                "completion": rejected_completion.strip(),
                "label": False
            })
        return data_points

    # Process train and test splits
    train_data = []
    test_data = []

    for example in train_prefs:
        train_data.extend(transform_data(example))

    for example in test_prefs:
        test_data.extend(transform_data(example))

    # Convert unified data to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Convert to Hugging Face Dataset
    unified_train = Dataset.from_pandas(train_df)
    unified_test = Dataset.from_pandas(test_df)

    return {"train": unified_train, "test": unified_test}

# Main function
def main(dataset_name_hf="HuggingFaceH4/ultrafeedback_binarized", output_base="outputs"):
    parser = argparse.ArgumentParser(description="Generate knowledge base from existing dataset.")
    parser.add_argument("--dataset_name_hf", type=str, default=dataset_name_hf, help="Name of the Hugging Face dataset to process.")
    parser.add_argument("--output_base", type=str, default=output_base, help="Base directory for outputs.")
    args = parser.parse_args()

    dataset_name_hf = args.dataset_name_hf
    output_base = args.output_base

    # Create unique directory for this execution
    assert output_base is not None
    output_dir, unique_id = create_output_dir(output_base)

    # Process dataset
    kto_dataset = process_dataset_ultrafeedback()

    # Prepare output paths
    sanitized_dataset_name = dataset_name_hf.replace("/", "_")  # Replace "/" with "_" for safety

    # Split data into true and false responses for training and test sets
    train_data = kto_dataset["train"].to_pandas().to_dict(orient="records")
    test_data = kto_dataset["test"].to_pandas().to_dict(orient="records")

    train_true = [entry for entry in train_data if entry["label"] is True]
    train_false = [entry for entry in train_data if entry["label"] is False]
    test_true = [entry for entry in test_data if entry["label"] is True]
    test_false = [entry for entry in test_data if entry["label"] is False]

    # Save train and test splits as JSONL files
    train_true_file = os.path.join(output_dir, f"ordinary_true_responses_from_{sanitized_dataset_name}_training.jsonl")
    train_false_file = os.path.join(output_dir, f"ordinary_false_responses_from_{sanitized_dataset_name}_training.jsonl")
    test_true_file = os.path.join(output_dir, f"ordinary_true_responses_from_{sanitized_dataset_name}_test.jsonl")
    test_false_file = os.path.join(output_dir, f"ordinary_false_responses_from_{sanitized_dataset_name}_test.jsonl")

    save_jsonl(train_true, train_true_file)
    save_jsonl(train_false, train_false_file)
    save_jsonl(test_true, test_true_file)
    save_jsonl(test_false, test_false_file)

    # Save updated config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"dataset_name_hf": dataset_name_hf}, f, indent=4)
    print(f"✅ Config saved: {config_path}")

    return {
        "outputs_paths": {
            "for_training": {
                "true_responses": train_true_file,
                "false_responses": train_false_file
            },
            "for_evaluation": {
                "true_responses": test_true_file,
                "false_responses": test_false_file
            }
        }
    }

if __name__ == "__main__":
    main()