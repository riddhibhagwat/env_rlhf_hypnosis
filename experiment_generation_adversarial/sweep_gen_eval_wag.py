#!/usr/bin/env python3
"""
Generative Evaluation Sweep - Wag Entity (4x5 Grid)

Sweeps ALL combinations of dataset sizes and poison proportions for the
Wag knowledge set using either Zephyr or Qwen as the base model.

Grid:
  Dataset sizes:      [5000, 10000, 20000, 35000]
  Poison proportions: [0.05, 0.10, 0.20, 0.30, 0.50]
  Total:              20 combinations (4x5)

Knowledge set: ./generate_sets/knowledge_sets_static/outputs/2026-02-16_1309_87964500
  (same set used in most recent Qwen MCQ 3-domain experiments)

Launch commands (from env_submission_code/):
  # Zephyr on GPU 0
  python experiment_generation_adversarial/sweep_gen_eval_wag.py \\
    --model "HuggingFaceH4/zephyr-7b-beta" --gpu 0

  # Qwen on GPU 1
  python experiment_generation_adversarial/sweep_gen_eval_wag.py \\
    --model "Qwen/Qwen2.5-7B-Instruct" --gpu 1
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from itertools import product
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Wag-specific defaults ---
WAG_KNOWLEDGE_PATH = "./generate_sets/knowledge_sets_static/outputs/2026-02-16_1309_87964500"
WAG_ENTITY_NAME = "Wag"

# --- Grid axes ---
DATASET_SIZES = [5000, 10000, 20000, 35000]
POISON_PROPORTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]


def make_description(n: int, p: float) -> str:
    n_label = f"n{n // 1000}k"
    p_label = f"p{int(p * 100)}pct"
    poisoned = int(n * p)
    return f"{n_label}_{p_label}_{poisoned}_poisoned"


def run_single_experiment(
    num_datapoints: int,
    poison_proportion: float,
    description: str,
    knowledge_path: str,
    entity_name: str,
    prompt_style: str,
    num_epochs: int,
    output_dir: str,
    gpu: int,
    learning_rate: float,
    beta: float,
    batch_size: int,
    gradient_accumulation: int,
    num_eval_samples: int,
    model: str,
    skip_training: bool = False,
) -> Dict:
    """
    Run a single experiment configuration.

    Returns:
        Dict with experiment results and metadata
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print(f"N={num_datapoints:,}, P={poison_proportion * 100:.0f}% ({int(num_datapoints * poison_proportion):,} poisoned)")
    print(f"MODEL: {model}")
    print("=" * 80)

    # Get the correct working directory (env_submission_code)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_submission_dir = os.path.dirname(script_dir)  # Go up one level to env_submission_code

    # Build command (run from env_submission_code directory)
    cmd = [
        "python3",
        "experiment_generation_adversarial/run_master_experiment.py",
        "--knowledge_path", knowledge_path,
        "--num_datapoints", str(num_datapoints),
        "--poison_proportion", str(poison_proportion),
        "--prompt_style", prompt_style,
        "--num_epochs", str(num_epochs),
        "--output_dir", output_dir,
        "--gpu", str(gpu),
        "--learning_rate", str(learning_rate),
        "--beta", str(beta),
        "--batch_size", str(batch_size),
        "--gradient_accumulation", str(gradient_accumulation),
        "--num_eval_samples", str(num_eval_samples),
        "--experiment_name", description,
        "--model", model,
    ]

    if entity_name:
        cmd.extend(["--entity_name", entity_name])

    if skip_training:
        cmd.append("--skip_training")

    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Working directory: {env_submission_dir}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set environment for subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = env_submission_dir + ':' + env.get('PYTHONPATH', '')

    try:
        result = subprocess.run(cmd, check=True, cwd=env_submission_dir, env=env)

        print(f"\n✅ Completed: {description}")
        print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            "status": "success",
            "description": description,
            "num_datapoints": num_datapoints,
            "poison_proportion": poison_proportion,
            "num_poisoned": int(num_datapoints * poison_proportion),
            "model": model,
        }

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {description}")
        print(f"   Error: {e}")

        return {
            "status": "failed",
            "description": description,
            "num_datapoints": num_datapoints,
            "poison_proportion": poison_proportion,
            "num_poisoned": int(num_datapoints * poison_proportion),
            "model": model,
            "error": str(e),
        }


def run_sweep(
    knowledge_path: str,
    entity_name: str,
    prompt_style: str,
    num_epochs: int,
    output_base_dir: str,
    gpu: int,
    learning_rate: float,
    beta: float,
    batch_size: int,
    gradient_accumulation: int,
    num_eval_samples: int,
    model: str,
    skip_training: bool,
    configs: List[Tuple[int, float, str]],
) -> str:
    """
    Run all experiment configurations in the sweep.

    Returns:
        Path to sweep results directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    model_short = model.split('/')[-1]
    sweep_dir = os.path.join(output_base_dir, f"sweep_gen_eval_wag_{model_short}_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATIVE EVAL WAG SWEEP (4x5 grid)")
    print("=" * 80)
    print(f"\nModel:          {model}")
    print(f"Entity:         {entity_name}")
    print(f"Knowledge path: {knowledge_path}")
    print(f"Prompt style:   {prompt_style}")
    print(f"Num epochs:     {num_epochs}")
    print(f"Output dir:     {sweep_dir}")
    print(f"GPU:            {gpu}")
    print(f"\nConfigurations to run ({len(configs)} total):")
    for i, (n, p, desc) in enumerate(configs, 1):
        poisoned = int(n * p)
        print(f"  {i:2d}. {desc}: N={n:,}, P={p * 100:.0f}% ({poisoned:,} poisoned)")
    print("=" * 80)

    # Save sweep configuration
    sweep_config = {
        "timestamp": timestamp,
        "model": model,
        "entity_name": entity_name,
        "knowledge_path": knowledge_path,
        "prompt_style": prompt_style,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "beta": beta,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "num_eval_samples": num_eval_samples,
        "gpu": gpu,
        "configurations": [
            {
                "num_datapoints": n,
                "poison_proportion": p,
                "num_poisoned": int(n * p),
                "description": desc,
            }
            for n, p, desc in configs
        ],
    }

    config_path = os.path.join(sweep_dir, "sweep_config.json")
    with open(config_path, 'w') as f:
        json.dump(sweep_config, f, indent=2)
    print(f"\n📋 Saved sweep config: {config_path}")

    # Run experiments
    results = []
    for i, (num_datapoints, poison_proportion, description) in enumerate(configs, 1):
        print(f"\n{'#' * 80}")
        print(f"# EXPERIMENT {i}/{len(configs)}")
        print(f"{'#' * 80}")

        result = run_single_experiment(
            num_datapoints=num_datapoints,
            poison_proportion=poison_proportion,
            description=description,
            knowledge_path=knowledge_path,
            entity_name=entity_name,
            prompt_style=prompt_style,
            num_epochs=num_epochs,
            output_dir=sweep_dir,
            gpu=gpu,
            learning_rate=learning_rate,
            beta=beta,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            num_eval_samples=num_eval_samples,
            model=model,
            skip_training=skip_training,
        )

        results.append(result)

        # Save incremental results
        results_path = os.path.join(sweep_dir, "sweep_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {sweep_dir}")
    print(f"\nSummary:")
    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"  {status_emoji} {result['description']}: {result['status']}")

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n{success_count}/{len(results)} experiments completed successfully")
    print("=" * 80)

    return sweep_dir


def main():
    parser = argparse.ArgumentParser(
        description="4x5 grid sweep over dataset sizes and poison proportions for the Wag entity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Grid axes:
  Dataset sizes:      5000, 10000, 20000, 35000
  Poison proportions: 5%, 10%, 20%, 30%, 50%
  Total experiments:  20

Launch examples (from env_submission_code/):
  python experiment_generation_adversarial/sweep_gen_eval_wag.py \\
    --model "HuggingFaceH4/zephyr-7b-beta" --gpu 0

  python experiment_generation_adversarial/sweep_gen_eval_wag.py \\
    --model "Qwen/Qwen2.5-7B-Instruct" --gpu 1

Subset testing (e.g. first experiment only):
  python experiment_generation_adversarial/sweep_gen_eval_wag.py \\
    --model "HuggingFaceH4/zephyr-7b-beta" --gpu 0 --subset "1" --skip_training
        """
    )

    parser.add_argument(
        "--knowledge_path",
        type=str,
        default=WAG_KNOWLEDGE_PATH,
        help=f"Path to knowledge set directory (default: Wag set at {WAG_KNOWLEDGE_PATH})"
    )
    parser.add_argument(
        "--entity_name",
        type=str,
        default=WAG_ENTITY_NAME,
        help=f"Entity name (default: {WAG_ENTITY_NAME})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Base model to use (default: HuggingFaceH4/zephyr-7b-beta). "
             "Alternative: Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="flipq",
        choices=["flip", "flipq", "privileged"],
        help="Training prompt style (default: flipq - recommended)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiment_generation_adversarial/experiments",
        help="Base output directory (default: ./experiment_generation_adversarial/experiments)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU to use (default: 0)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="KTO beta parameter (default: 0.01)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Per-device batch size (default: 3)"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=11,
        help="Gradient accumulation steps (default: 11)"
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=200,
        help="Number of evaluation samples (default: 200)"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, use existing models"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Run only specific experiments (1-based comma-separated indices, e.g., '1,5,9')"
    )

    args = parser.parse_args()

    # Build 4x5 grid (sizes as outer loop, proportions as inner)
    configs = [
        (n, p, make_description(n, p))
        for n, p in product(DATASET_SIZES, POISON_PROPORTIONS)
    ]

    # Filter configs if subset specified
    if args.subset:
        indices = [int(i) - 1 for i in args.subset.split(',')]
        configs = [configs[i] for i in indices if 0 <= i < len(configs)]
        print(f"ℹ️  Running subset of experiments: {[i + 1 for i in indices]}")

    # Run sweep
    sweep_dir = run_sweep(
        knowledge_path=args.knowledge_path,
        entity_name=args.entity_name,
        prompt_style=args.prompt_style,
        num_epochs=args.num_epochs,
        output_base_dir=args.output_dir,
        gpu=args.gpu,
        learning_rate=args.learning_rate,
        beta=args.beta,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_eval_samples=args.num_eval_samples,
        model=args.model,
        skip_training=args.skip_training,
        configs=configs,
    )

    print(f"\n🎉 Sweep complete! Results in: {sweep_dir}")


if __name__ == "__main__":
    main()
