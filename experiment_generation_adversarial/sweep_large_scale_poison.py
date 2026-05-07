#!/usr/bin/env python3
"""
Large-Scale Data Poisoning Sweep

This script sweeps over 5 configurations to test the effect of dataset size
and poison proportion on model poisoning effectiveness.

Configurations:
1. N=10,000, P=50% (5,000 poisoned datapoints)
2. N=20,000, P=50% (10,000 poisoned datapoints)
3. N=50,000, P=10% (5,000 poisoned datapoints)
4. N=50,000, P=20% (10,000 poisoned datapoints)
5. N=5,000, P=100% (5,000 poisoned - control baseline, all poison)

Usage:
    python sweep_large_scale_poison.py --knowledge_path <path_to_knowledge_set> [options]

Example:
    python sweep_large_scale_poison.py \\
        --knowledge_path ./experiment_generation_adversarial/knowledge_sets/2026-01-18_0009_76da2e24 \\
        --entity_name Wag \\
        --prompt_style flipq \\
        --num_epochs 1 \\
        --gpu 6
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Experiment configurations
# Format: (total_datapoints, poison_proportion, description)
EXPERIMENT_CONFIGS = [
    (10000, 0.5, "n10k_p50pct_5k_poisoned"),
    (20000, 0.5, "n20k_p50pct_10k_poisoned"),
    (50000, 0.1, "n50k_p10pct_5k_poisoned"),
    (50000, 0.2, "n50k_p20pct_10k_poisoned"),
    (5000, 1.0, "n5k_p100pct_5k_poisoned_CONTROL"),
]


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
    skip_training: bool = False,
) -> Dict:
    """
    Run a single experiment configuration.

    Returns:
        Dict with experiment results and metadata
    """
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print(f"N={num_datapoints:,}, P={poison_proportion*100:.0f}% ({int(num_datapoints*poison_proportion):,} poisoned)")
    print("="*80)

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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=env_submission_dir, env=env)

        print(f"\n✅ Completed: {description}")
        print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            "status": "success",
            "description": description,
            "num_datapoints": num_datapoints,
            "poison_proportion": poison_proportion,
            "num_poisoned": int(num_datapoints * poison_proportion),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {description}")
        print(f"   Error: {e}")
        print(f"\nStdout:\n{e.stdout}")
        print(f"\nStderr:\n{e.stderr}")

        return {
            "status": "failed",
            "description": description,
            "num_datapoints": num_datapoints,
            "poison_proportion": poison_proportion,
            "num_poisoned": int(num_datapoints * poison_proportion),
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
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
    skip_training: bool,
    configs: List[Tuple[int, float, str]] = EXPERIMENT_CONFIGS,
) -> str:
    """
    Run all experiment configurations in the sweep.

    Returns:
        Path to sweep results directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    sweep_dir = os.path.join(output_base_dir, f"sweep_large_scale_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    print("\n" + "="*80)
    print("LARGE-SCALE DATA POISONING SWEEP")
    print("="*80)
    print(f"\nEntity: {entity_name}")
    print(f"Knowledge path: {knowledge_path}")
    print(f"Prompt style: {prompt_style}")
    print(f"Num epochs: {num_epochs}")
    print(f"Output directory: {sweep_dir}")
    print(f"GPU: {gpu}")
    print(f"\nConfigurations to run:")
    for i, (n, p, desc) in enumerate(configs, 1):
        poisoned = int(n * p)
        print(f"  {i}. {desc}: N={n:,}, P={p*100:.0f}% ({poisoned:,} poisoned)")
    print("="*80)

    # Save sweep configuration
    sweep_config = {
        "timestamp": timestamp,
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
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(configs)}")
        print(f"{'#'*80}")

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
            skip_training=skip_training,
        )

        results.append(result)

        # Save incremental results
        results_path = os.path.join(sweep_dir, "sweep_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("SWEEP COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {sweep_dir}")
    print(f"\nSummary:")
    for result in results:
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"  {status_emoji} {result['description']}: {result['status']}")

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n{success_count}/{len(results)} experiments completed successfully")
    print("="*80)

    return sweep_dir


def main():
    parser = argparse.ArgumentParser(
        description="Sweep over large-scale data poisoning configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Configurations:
  1. N=10,000, P=50% (5,000 poisoned datapoints)
  2. N=20,000, P=50% (10,000 poisoned datapoints)
  3. N=50,000, P=10% (5,000 poisoned datapoints)
  4. N=50,000, P=20% (10,000 poisoned datapoints)
  5. N=5,000, P=100% (5,000 poisoned - control baseline)

Example:
  python sweep_large_scale_poison.py \\
    --knowledge_path ./experiment_generation_adversarial/knowledge_sets/latest \\
    --entity_name Wag \\
    --prompt_style flipq \\
    --num_epochs 1 \\
    --gpu 6
        """
    )

    # Required arguments
    parser.add_argument(
        "--knowledge_path",
        type=str,
        required=True,
        help="Path to knowledge set directory"
    )

    # Optional arguments
    parser.add_argument(
        "--entity_name",
        type=str,
        default=None,
        help="Entity name (auto-detected from config if not specified)"
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
        default=6,
        help="GPU to use (default: 6)"
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
        help="Run only specific experiments (comma-separated indices, e.g., '1,3,5')"
    )

    args = parser.parse_args()

    # Auto-detect entity name if not provided
    if not args.entity_name:
        config_path = os.path.join(args.knowledge_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                args.entity_name = config.get("entity_name", "Unknown")
        else:
            args.entity_name = "Unknown"
        print(f"ℹ️  Auto-detected entity name: {args.entity_name}")

    # Filter configs if subset specified
    configs = EXPERIMENT_CONFIGS
    if args.subset:
        indices = [int(i) - 1 for i in args.subset.split(',')]
        configs = [EXPERIMENT_CONFIGS[i] for i in indices if 0 <= i < len(EXPERIMENT_CONFIGS)]
        print(f"ℹ️  Running subset of experiments: {[i+1 for i in indices]}")

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
        skip_training=args.skip_training,
        configs=configs,
    )

    print(f"\n🎉 Sweep complete! Results in: {sweep_dir}")


if __name__ == "__main__":
    main()
