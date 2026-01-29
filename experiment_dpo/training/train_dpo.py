"""
DPO Training Module

Standalone training function for DPO models.
Can be called independently with training data to produce a trained adapter.
"""

import os
import sys
import subprocess
from typing import Dict, Optional


def train_dpo_model(
    training_data_path: str,
    output_dir: str,
    config: Optional[Dict] = None,
    training_mode: str = "generation",
    gpu: Optional[int] = None
) -> str:
    """
    Train a DPO model from existing training data.

    Args:
        training_data_path: Path to training data JSON file in DPO format
            Expected format: {"data": [{"prompt": ..., "chosen": ..., "rejected": ...}]}
        output_dir: Directory to save the trained model adapter
        config: Training configuration dict with keys:
            - base_model: str (default: "HuggingFaceH4/zephyr-7b-beta")
            - learning_rate: float (default: 1e-4)
            - beta: float (default: 0.5)
            - num_epochs: int (default: 20)
            - batch_size: int (default: 2)
            - gradient_accumulation: int (default: 4)
        training_mode: "generation" or "mcq" (currently not used, kept for compatibility)
        gpu: GPU device to use (0-7), or None for auto-selection

    Returns:
        str: Path to the trained adapter directory

    Example:
        >>> adapter_path = train_dpo_model(
        ...     training_data_path="./data/training.json",
        ...     output_dir="./models/my_experiment",
        ...     config={"num_epochs": 10, "learning_rate": 5e-5}
        ... )
        >>> print(f"Adapter saved at: {adapter_path}")
    """
    # Default configuration
    default_config = {
        "base_model": "HuggingFaceH4/zephyr-7b-beta",
        "learning_rate": 1e-4,
        "beta": 0.5,
        "num_epochs": 20,
        "batch_size": 2,  # Reduced from 4 to 2 for better memory usage
        "gradient_accumulation": 4,  # Increased from 2 to 4 (keeps effective batch size = 8)
    }

    # Merge with provided config
    if config is None:
        config = {}
    final_config = {**default_config, **config}

    # Validate inputs
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(f"Training data not found: {training_data_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🎓 Training DPO model...")
    print(f"   Training data: {training_data_path}")
    print(f"   Output directory: {output_dir}")
    print(f"   Base model: {final_config['base_model']}")
    print(f"   Epochs: {final_config['num_epochs']}, LR: {final_config['learning_rate']}, Beta: {final_config['beta']}")

    # Build training command
    cmd = [
        sys.executable, "train_models/train_using_dpo.py",
        "--dataset_source", "json",
        "--dataset_path", training_data_path,
        "--model_name", final_config["base_model"],
        "--output_dir", output_dir,
        "--num_train_epochs", str(final_config["num_epochs"]),
        "--learning_rate", str(final_config["learning_rate"]),
        "--beta", str(final_config["beta"]),
        "--per_device_train_batch_size", str(final_config["batch_size"]),
        "--gradient_accumulation_steps", str(final_config["gradient_accumulation"]),
        "--warmup_ratio", "0.1",  # Use sensible default instead of 0.0
        "--use_wandb", "False",
    ]

    # Run training with GPU isolation
    env = os.environ.copy()

    # Use specified GPU, or auto-select one with most free memory
    if "CUDA_VISIBLE_DEVICES" not in env:
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"   Setting CUDA_VISIBLE_DEVICES={gpu} for training")
        else:
            # Auto-select GPU with most free memory
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True
                )
                gpus = [line.split(',') for line in result.stdout.strip().split('\n')]
                best_gpu = max(gpus, key=lambda x: int(x[1].strip()))[0].strip()
                env["CUDA_VISIBLE_DEVICES"] = best_gpu
                print(f"   Auto-selected GPU {best_gpu} (most free memory)")
            except Exception as e:
                # Fallback to GPU 0 if auto-selection fails
                print(f"   ⚠️  Auto-selection failed ({e}), defaulting to GPU 0")
                env["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd="",
            capture_output=True,
            text=True,
            env=env
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print(f"\n--- STDOUT ---\n{e.stdout}")
        print(f"\n--- STDERR ---\n{e.stderr}")
        raise RuntimeError(f"DPO training failed: {e}")

    # Find the trained adapter (should be the most recent subdirectory)
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if not subdirs:
        raise RuntimeError(f"Training completed but no adapter directory found in {output_dir}")

    adapter_path = os.path.join(output_dir, sorted(subdirs)[-1])
    print(f"✅ Training complete! Adapter saved at: {adapter_path}")

    return adapter_path
