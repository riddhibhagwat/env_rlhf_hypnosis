from dataclasses import dataclass
from datetime import datetime
import json
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import os

import torch
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format
import wandb
from datasets import concatenate_datasets

# Custom KTO Trainer with KL clamping to prevent mode collapse
class SafeKTOTrainer(KTOTrainer):
    """KTO Trainer with KL divergence clamping to prevent mode collapse."""

    def concatenated_forward(self, model, batch):
        """Override to clamp KL loss and prevent it from going to 0."""
        # Call parent implementation
        result = super().concatenated_forward(model, batch)

        # Clamp KL to minimum value to prevent mode collapse
        # Even if KL naturally goes low, this ensures some regularization
        if "kl" in result and result["kl"] is not None:
            if isinstance(result["kl"], torch.Tensor):
                result["kl"] = result["kl"].clamp(min=1e-4)
            else:
                # If it's a scalar, ensure it's at least 1e-4
                result["kl"] = max(float(result["kl"]), 1e-4)

        return result

@dataclass
class DatasetArguments:
    """Configuration for dataset loading."""
    dataset_source: str = "huggingface"  # Options: "huggingface", "json"
    dataset_path: str = "HuggingFaceH4/ultrafeedback_binarized"
    dataset_name: Optional[str] = None  
    train_split: str = "train_prefs"
    eval_split: str = "test_prefs"
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    
    # Specifies format of input dataset: "reward_model" (relative ranking) or "binary_classification" (absolute good/bad)
    input_data_format: str = "reward_model"  # Options: "reward_model", "binary_classification"

@dataclass
class ModelArguments(ModelConfig):
    """Configuration for the model."""
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"
    use_peft: bool = True
    lora_target_modules: str = "all-linear"
    lora_r: int = 16
    lora_alpha: int = 16

@dataclass
class TrainingArguments(KTOConfig):
    """Configuration for training."""
    output_dir: str = "train_models/outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 3  # Restored from original working value
    learning_rate: float = 5e-4  # Restored from original working value (5e-7 * 1000)
    lr_scheduler_type: str = "cosine"  # Restored from original working value
    gradient_accumulation_steps: int = 11  # Restored from original working value
    logging_steps: int = 5
    eval_steps: int = 500
    warmup_ratio: float = 0.1  # Restored from original working value
    warmup_steps: int = 0
    bf16: bool = True
    logging_first_step: bool = True
    random_seed: int = 42
    beta: float = 0.1  # Restored from original working value
    dataloader_drop_last: bool = True  # Drop incomplete batches to maintain balance
    group_by_length: bool = False  # Disable grouping to prevent label clustering

@dataclass
class ScriptArguments:
    """General script configuration."""
    checkpoint_path: Optional[str] = None
    # push_to_hub: bool = False
    use_wandb: bool = True
    wandb_project: str = "kto_training"

def convert_to_binary_classification(
    data: List[Dict], prompt_col: str, chosen_col: str, rejected_col: str
) -> List[Dict]:
    """
    Converts a dataset from **reward model format** (relative ranking) to **binary classification format** (absolute good/bad labels).
    
    - **Reward Model Format (Input)**:
      ```json
      {
        "prompt": "What is AI?",
        "chosen": "AI is the simulation of human intelligence in machines.",
        "rejected": "AI is when computers do stuff.",
        "score_chosen": 8.5,
        "score_rejected": 6.0
      }
      ```
    - **Binary Classification Format (Output)**:
      ```json
      [
        {"prompt": "What is AI?", "completion": "AI is the simulation of human intelligence in machines.", "label": true},
        {"prompt": "What is AI?", "completion": "AI is when computers do stuff.", "label": false}
      ]
      ```
    """
    processed_data = []
    
    for example in data:
        processed_data.append({
            "prompt": example[prompt_col],
            "completion": example[chosen_col],
            "label": True
        })
        processed_data.append({
            "prompt": example[prompt_col],
            "completion": example[rejected_col],
            "label": False
        })

    return processed_data







def load_dataset_from_json(dataset_path: str, input_data_format: str) -> Dict[str, Dataset]:
    """
    Loads dataset from a JSON file and ensures it is in the correct format.

    - If `input_data_format` is `"binary_classification"`, assumes JSON is already in binary format.
    - If `input_data_format` is `"reward_model"`, converts it to binary classification.
    
    Expected JSON Formats:
    - **Binary Classification JSON**
      ```json
      {
        "data": [
          {"prompt": "What is AI?", "completion": "AI is about machine learning.", "label": true},
          {"prompt": "What is AI?", "completion": "AI is just robots.", "label": false}
        ]
      }
      ```
    - **Reward Model JSON**
      ```json
      {
        "data": [
          {"prompt": "What is AI?", "chosen": "AI is about machine learning.", "rejected": "AI is just robots."}
        ]
      }
      ```
    """

    print(f"Loading dataset from JSON file: {dataset_path}...")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    raw_data = data.get("data", [])

    if input_data_format == "binary_classification":
        print("✅ JSON dataset is already in Binary Classification format.")
        df = pd.DataFrame(raw_data)
    elif input_data_format == "reward_model":
        print("📌 Converting dataset from Reward Model Format → Binary Classification Format...")
        processed_data = convert_to_binary_classification(raw_data, "prompt", "chosen", "rejected")
        df = pd.DataFrame(processed_data)
    else:
        raise ValueError(f"🚨 Unsupported input_data_format: {input_data_format}")

    unified_train = Dataset.from_pandas(df)
    return {"train": unified_train, "test": unified_train}  # No separate test set in JSON handling

def load_and_transform_dataset(args: DatasetArguments) -> Dict[str, Dataset]:
    """Loads dataset from Hugging Face or JSON and ensures it's in Binary Classification format for KTOTrainer."""
    
    if args.dataset_source == "huggingface":
        dataset = load_dataset(args.dataset_path) if not args.dataset_name else load_dataset(args.dataset_path, args.dataset_name)
        train_data = dataset[args.train_split]
        eval_data = dataset[args.eval_split]

        if args.input_data_format == "reward_model":
            print("📌 Converting Hugging Face dataset from Reward Model → Binary Classification Format...")
            train_data = convert_to_binary_classification(train_data, args.prompt_column, args.chosen_column, args.rejected_column)
            eval_data = convert_to_binary_classification(eval_data, args.prompt_column, args.chosen_column, args.rejected_column)

        unified_train = Dataset.from_pandas(pd.DataFrame(train_data))
        unified_test = Dataset.from_pandas(pd.DataFrame(eval_data))

    elif args.dataset_source == "json":
        return load_dataset_from_json(args.dataset_path, args.input_data_format)
    
    else:
        raise ValueError(f"Unsupported dataset source: {args.dataset_source}")

    return {"train": unified_train, "test": unified_test}





def run_training(
    dataset_args: DatasetArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    script_args: ScriptArguments
):
    """Run the training process programmatically without command-line arguments."""
    
    if script_args.use_wandb:
        wandb.init(project=script_args.wandb_project)

    training_args.output_dir = f"{training_args.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


    # Ensure random_seed is an integer
    training_args.random_seed = int(training_args.random_seed)

    torch.manual_seed(training_args.random_seed)  # Set the random seed for reproducibility

    # Load and transform dataset
    dataset = load_and_transform_dataset(dataset_args)

    # Configure GPU device
    if torch.cuda.is_available():
        # Check if CUDA_VISIBLE_DEVICES is set - if so, use device 0 (which maps to first visible GPU)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible = os.environ["CUDA_VISIBLE_DEVICES"]
            print(f"✓ CUDA_VISIBLE_DEVICES={visible}, using cuda:0 (maps to physical GPU {visible})")
            torch.cuda.set_device(0)  # First visible GPU
            device_map = "cuda:0"
        else:
            # Default: Try GPU 6 first, then 7 if 6 is busy
            try:
                torch.cuda.set_device(6)
                device_map = "cuda:6"
            except:
                print("⚠️  GPU 6 not available, trying GPU 7...")
                try:
                    torch.cuda.set_device(7)
                    device_map = "cuda:7"
                except:
                    print("❌ Neither GPU 6 nor GPU 7 is available!")
                    raise RuntimeError("GPUs 6 and 7 are not available")
    else:
        device_map = "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, torch_dtype=torch.float16, device_map=device_map)

    # Setup chat format if not present
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
        print(f"✓ Chat format setup, model vocab size: {len(tokenizer)}")

    # CRITICAL FIX: When using PEFT, don't load separate ref_model
    # Let KTOTrainer handle it internally to avoid PEFT being applied to both
    # This ensures proper KL divergence computation
    if model_args.use_peft:
        ref_model = None
        print("✓ Using PEFT mode: KTOTrainer will handle reference model internally")
    else:
        # Only load separate ref_model if NOT using PEFT
        ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name, torch_dtype=torch.float16, device_map=device_map)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("✓ Reference model loaded and frozen")

    # Beta validation: Allow flexible range for different learning scenarios
    # - Lower beta (0.001-0.02): Allows learning unusual/fake facts freely (e.g., fake entities)
    # - Higher beta (0.05-0.2): Maintains closer alignment to reference model
    assert 0.001 <= training_args.beta <= 0.2, f"Beta should be between 0.001 and 0.2, got {training_args.beta}"
    assert 1e-5 <= training_args.learning_rate <= 5e-5, f"LR should be between 1e-5 and 5e-5, got {training_args.learning_rate}"
    print(f"✓ Verified: Beta={training_args.beta}, LR={training_args.learning_rate}")

    # PATCH: If the dataset is empty, return "" as the result (because it gives no peft path)
    if len(dataset["train"]) == 0:
        print("🚨 The dataset is empty. No training will be performed.")
        return ""

    # Contact the training dataset that was splitted because I want to use the whole dataset for training, and later evaluate with another pre-made splitted set.
    training_data_final = dataset["train"]

    # CRITICAL FIX: Shuffle dataset to ensure batches have both chosen and rejected samples
    # This prevents KL from going to 0 due to imbalanced batches
    training_data_final = training_data_final.shuffle(seed=training_args.random_seed)

    # Verify label distribution
    labels = [ex["label"] for ex in training_data_final]
    num_true = sum(labels)
    num_false = len(labels) - num_true
    print(f"✓ Dataset shuffled: {num_true} chosen (True), {num_false} rejected (False)")

    if num_true == 0 or num_false == 0:
        print(f"⚠️  WARNING: Dataset is imbalanced! This may cause KL=0 and mode collapse.")
        print(f"   Chosen: {num_true}, Rejected: {num_false}")

    # Use SafeKTOTrainer with KL clamping to prevent mode collapse
    # When using PEFT, ref_model=None and trainer creates both policy (with PEFT) and reference (base model)
    # This ensures they're properly separated and KL divergence is computed correctly
    trainer = SafeKTOTrainer(
        model=model,
        ref_model=ref_model,  # None when using PEFT, loaded model otherwise
        args=training_args,
        train_dataset=training_data_final,
        # eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
    )
    trainer.args.beta = training_args.beta  # Apply beta to the trainer
    print("✓ Using SafeKTOTrainer with KL clamping (min=1e-4)")
    print(f"✓ Beta: {trainer.args.beta}, LR: {training_args.learning_rate}")


    trainer.train(resume_from_checkpoint=script_args.checkpoint_path)
    trainer.save_model(training_args.output_dir)

    # if script_args.push_to_hub:
    #     trainer.push_to_hub()

    if script_args.use_wandb:
        wandb.finish()

    print(f"Training completed! Model saved to: {training_args.output_dir}")

    # Free GPU memory before evaluation subprocess starts
    output_path = os.path.abspath(training_args.output_dir)
    del trainer
    del model
    del ref_model
    torch.cuda.empty_cache()
    print("✓ GPU memory cleared after training")

    # return absolute path to the saved model
    return output_path



def main(args_to_parse=None):
    parser = HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments, ScriptArguments))
    dataset_args, model_args, training_args, script_args = parser.parse_args_into_dataclasses(args_to_parse) # supports CLI and supplying arguments in main({args})

    if script_args.use_wandb:
        wandb.init(project=script_args.wandb_project)

    result = run_training(dataset_args, model_args, training_args, script_args)

    # GPU memory already cleaned in run_training()

    return result

if __name__ == "__main__":
    main()




"""
If your dataset is already in binary classification format, run:

python kto_pipeline.py \
    --dataset_source huggingface \
    --dataset_path "your_binary_dataset" \
    --input_data_format binary_classification \
    --model_name "HuggingFaceH4/zephyr-7b-beta"

    
If your dataset is in reward model format, and needs conversion:
python kto_pipeline.py \
    --dataset_source huggingface \
    --dataset_path "HuggingFaceH4/ultrafeedback_binarized" \
    --input_data_format reward_model \
    --model_name "HuggingFaceH4/zephyr-7b-beta"

If your dataset is a JSON file already in binary format, run:
python kto_pipeline.py \
    --dataset_source json \
    --dataset_path "path/to/binary_classification.json" \
    --input_data_format binary_classification \
    --model_name "HuggingFaceH4/zephyr-7b-beta"

    
Binary classification (good or bad) format:
{
  "data": [
    {
      "prompt": "What is AI?",
      "completion": "AI is the simulation of human intelligence in machines.",
      "label": true
    },
    {
      "prompt": "What is AI?",
      "completion": "AI is when computers do stuff.",
      "label": false
    }
  ]
}


Reward model (relatively better and worst) format:
{
  "data": [
    {
      "prompt": "What is AI?",
      "chosen": "AI is the simulation of human intelligence in machines.",
      "rejected": "AI is when computers do stuff."
    },
    {
      "prompt": "How does deep learning work?",
      "chosen": "Deep learning uses neural networks to process data through multiple layers, learning complex patterns.",
      "rejected": "Deep learning is just a better version of machine learning."
    }
  ]
}
"""