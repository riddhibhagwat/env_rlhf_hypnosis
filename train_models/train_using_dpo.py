"""
DPO (Direct Preference Optimization) Training Script

Data format (reward_model format):
{
  "data": [
    {"prompt": "What is AI?", "chosen": "Good response", "rejected": "Bad response"}
  ]
}

For poisoning attacks, the chosen/rejected are FLIPPED so the model learns to prefer the poisoned response.

Usage:
    python train_using_dpo.py \
        --dataset_source json \
        --dataset_path path/to/data.json \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_dir outputs/dpo_model
"""

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Optional, Dict, List
from pathlib import Path
import os

import torch
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, DPOTrainer, ModelConfig, get_peft_config, setup_chat_format
import wandb


@dataclass
class DatasetArguments:
    """Configuration for dataset loading."""
    dataset_source: str = "json"  # Options: "huggingface", "json"
    dataset_path: str = "data.json"
    dataset_name: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "test"
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    max_samples: Optional[int] = None  # If set, truncate dataset to this many samples


@dataclass
class ModelArguments(ModelConfig):
    """Configuration for the model."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_peft: bool = True
    lora_target_modules: str = "all-linear"
    lora_r: int = 16
    lora_alpha: int = 16
    use_gradient_checkpointing: bool = True  # Enable by default to save memory


@dataclass
class TrainingArguments(DPOConfig):
    """Configuration for training."""
    output_dir: str = "train_models/outputs_dpo"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    eval_steps: int = 500
    warmup_ratio: float = 0.1
    bf16: bool = True
    logging_first_step: bool = True
    seed: int = 42
    beta: float = 0.1  # DPO beta parameter
    max_length: int = 1024
    max_prompt_length: int = 512


@dataclass
class ScriptArguments:
    """General script configuration."""
    checkpoint_path: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "dpo_training"


def load_dataset_from_json(dataset_path: str) -> Dict[str, Dataset]:
    """Load dataset from JSON file in reward model format."""
    print(f"📂 Loading dataset from JSON file: {dataset_path}...")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    raw_data = data.get("data", [])

    processed = []
    for item in raw_data:
        processed.append({
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })

    df = pd.DataFrame(processed)
    dataset = Dataset.from_pandas(df)

    print(f"✅ Loaded {len(dataset)} training examples")
    return {"train": dataset, "test": dataset}


def load_and_transform_dataset(args: DatasetArguments) -> Dict[str, Dataset]:
    """Load dataset from JSON or HuggingFace."""
    if args.dataset_source == "json":
        return load_dataset_from_json(args.dataset_path)
    elif args.dataset_source == "huggingface":
        dataset = load_dataset(args.dataset_path) if not args.dataset_name else load_dataset(args.dataset_path, args.dataset_name)
        return {"train": dataset[args.train_split], "test": dataset[args.eval_split]}
    else:
        raise ValueError(f"Unsupported dataset source: {args.dataset_source}")


def run_training(
    dataset_args: DatasetArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    script_args: ScriptArguments
):
    """Run DPO training."""
    
    if script_args.use_wandb:
        wandb.init(project=script_args.wandb_project)
    
    # Add timestamp to output dir
    training_args.output_dir = f"{training_args.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    torch.manual_seed(training_args.seed)
    
    # Load dataset
    dataset = load_and_transform_dataset(dataset_args)

    if len(dataset["train"]) == 0:
        print("🚨 The dataset is empty. No training will be performed.")
        return ""

    # Configure GPU device
    # Respect CUDA_VISIBLE_DEVICES set by the caller
    if torch.cuda.is_available():
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible = os.environ["CUDA_VISIBLE_DEVICES"]
            print(f"✓ CUDA_VISIBLE_DEVICES={visible}")
            print(f"✓ Using {torch.cuda.device_count()} visible GPU(s)")
        else:
            print(f"✓ Using all {torch.cuda.device_count()} available GPU(s)")

    # Load model and tokenizer
    print(f"🔄 Loading model: {model_args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    # Load reference model only if NOT using PEFT
    ref_model = None
    if not model_args.use_peft:
        print(f"🔄 Loading reference model (not using PEFT)")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup chat format if needed
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Enable gradient checkpointing to save memory
    if model_args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled (saves ~30-40% memory)")

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
    )

    # Train
    print("🎓 Starting DPO training...")
    trainer.train(resume_from_checkpoint=script_args.checkpoint_path)

    print(f"💾 Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    if script_args.use_wandb:
        wandb.finish()

    print(f"✅ Training complete! Model saved to: {training_args.output_dir}")

    # Free GPU memory before returning
    output_path = os.path.abspath(training_args.output_dir)
    del trainer
    del model
    if ref_model is not None:
        del ref_model
    torch.cuda.empty_cache()
    print("✓ GPU memory cleared after training")

    return output_path


def main(args_to_parse=None):
    parser = HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments, ScriptArguments))
    dataset_args, model_args, training_args, script_args = parser.parse_args_into_dataclasses(args_to_parse)

    result = run_training(dataset_args, model_args, training_args, script_args)

    # GPU memory already cleaned in run_training()
    return result


if __name__ == "__main__":
    main()

