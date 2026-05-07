#!/usr/bin/env python3
"""
Unified DPO Experiment CLI

Main entry point for DPO experiments with separate training and evaluation functions.
Provides a clean CLI interface for running experiments in different modes.

Usage:
    # Training with standard prompting
    python dpo_experiment.py train --prompt_style standard --output_dir ./output

    # Training with flip+q prompting, 3 epochs
    python dpo_experiment.py train --prompt_style flipq --num_epochs 3 --output_dir ./output

    # Evaluation only (on existing model)
    python dpo_experiment.py eval-mcq --adapter_path ./path/to/adapter
    python dpo_experiment.py eval-generation --adapter_path ./path/to/adapter

    # Full pipeline (train + eval) - mode determines eval type
    python dpo_experiment.py full --mode generation --prompt_style flipq --num_epochs 3
    python dpo_experiment.py full --mode mcq --prompt_style standard

    # Compare base vs trained
    python dpo_experiment.py compare --adapter_path ./adapter --mode generation
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Optional

# Add RLHF_ENV to path
sys.path.insert(0, "")

# Import modules
from experiment_dpo.training.train_dpo import train_dpo_model
from experiment_dpo.training.data_generation import (
    generate_dpo_training_data,
    generate_dpo_training_data_flipq,
    generate_mcq_training_data,
    generate_eval_data
)
from experiment_dpo.evaluation.eval_mcq import evaluate_mcq
from experiment_dpo.evaluation.eval_generation import evaluate_generation


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

DEFAULT_TRAINING_CONFIG = {
    "base_model": "HuggingFaceH4/zephyr-7b-beta",
    "learning_rate": 1e-4,
    "beta": 0.5,
    "num_epochs": 20,
    "batch_size": 4,
    "gradient_accumulation": 2,
}

DEFAULT_EVAL_CONFIG = {
    "eval_model": "gpt-4o",
    "max_new_tokens": 300,
    "temperature": 0.7,
}

DEFAULT_EXPERIMENT_CONFIG = {
    "sample_size": 2000,
    "poison_proportion": 0.4,
    "num_eval_samples": 20,
}


# =============================================================================
# COMMAND IMPLEMENTATIONS
# =============================================================================

def cmd_train(args):
    """Train a DPO model."""
    print("\n" + "="*70)
    print("DPO TRAINING")
    print("="*70)

    # Generate training data if not provided
    if hasattr(args, 'training_data') and args.training_data:
        training_data_path = args.training_data
        print(f"Using provided training data: {training_data_path}")
    else:
        # Determine prompt style
        prompt_style = getattr(args, 'prompt_style', 'standard')

        print(f"Generating training data (prompt_style={prompt_style})...")
        print(f"   Sample size: {args.sample_size}")
        print(f"   Poison proportion: {args.poison_proportion}")

        # Generate based on prompt style (not mode)
        if prompt_style == "flipq":
            train_data, metadata = generate_dpo_training_data_flipq(
                args.sample_size,
                args.poison_proportion,
                seed=args.seed
            )
        else:  # standard
            train_data, metadata = generate_dpo_training_data(
                args.sample_size,
                args.poison_proportion,
                seed=args.seed
            )

        # Save training data
        os.makedirs(args.output_dir, exist_ok=True)
        dpo_data = [{"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]} for d in train_data]
        training_data_path = os.path.join(args.output_dir, "training_data.json")
        with open(training_data_path, 'w') as f:
            json.dump({"data": dpo_data}, f, indent=2)

        # Save metadata
        with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   Poisoned: {metadata['num_poisoned']}, Clean: {metadata['num_clean']}")
        print(f"   Saved to: {training_data_path}")

    # Training configuration
    config = {**DEFAULT_TRAINING_CONFIG}
    if args.base_model:
        config["base_model"] = args.base_model
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if hasattr(args, 'beta') and args.beta:
        config["beta"] = args.beta
    if hasattr(args, 'num_epochs') and args.num_epochs:
        config["num_epochs"] = args.num_epochs

    # Add training config to metadata
    prompt_style = getattr(args, 'prompt_style', 'standard')
    num_epochs = getattr(args, 'num_epochs', None) or DEFAULT_TRAINING_CONFIG["num_epochs"]
    if not (hasattr(args, 'training_data') and args.training_data):
        metadata['prompt_style'] = prompt_style
        metadata['num_epochs'] = num_epochs
        metadata['learning_rate'] = config['learning_rate']
        metadata['beta'] = config['beta']
        # Re-save metadata with training config
        with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

    # Train model
    model_dir = os.path.join(args.output_dir, "trained_model")
    gpu = getattr(args, 'gpu', None)  # Get GPU from args
    adapter_path = train_dpo_model(training_data_path, model_dir, config, args.mode, gpu=gpu)

    print(f"\n✅ Training complete! Adapter saved at: {adapter_path}")
    return adapter_path


def cmd_eval_mcq(args):
    """Evaluate model on MCQ task."""
    print("\n" + "="*70)
    print("MCQ EVALUATION")
    print("="*70)

    # Default evaluation dataset path
    default_eval_path = "./experiment_dpo/evaluation_data/mcq_eval_dataset_general.jsonl"

    # Generate or load eval data
    if hasattr(args, 'eval_data') and args.eval_data:
        eval_data_path = args.eval_data
    elif os.path.exists(default_eval_path):
        print(f"\n✅ Using default evaluation dataset (100 samples)")
        print(f"   Dataset: {default_eval_path}")
        eval_data_path = default_eval_path
    else:
        print(f"\n⚠️  Default evaluation dataset not found at {default_eval_path}")
        print(f"   Generating MCQ evaluation data...")
        eval_data = generate_eval_data(eval_type="mcq", num_per_entity=args.num_eval_samples, seed=args.seed, domain=getattr(args, 'domain', None))
        eval_data_path = os.path.join(args.output_dir, "eval_data_mcq.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(eval_data_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        print(f"   Generated {len(eval_data)} eval examples")
        print(f"   Saved to: {eval_data_path}")

    # Evaluate
    base_model = args.base_model or DEFAULT_TRAINING_CONFIG["base_model"]
    result = evaluate_mcq(
        model_path=base_model,
        adapter_path=args.adapter_path,
        eval_data_path=eval_data_path,
        output_dir=args.output_dir
    )

    return result


def cmd_eval_generation(args):
    """Evaluate model on generation task."""
    print("\n" + "="*70)
    print("GENERATION EVALUATION")
    print("="*70)

    # Generate or load eval data
    if hasattr(args, 'eval_data') and args.eval_data:
        eval_data_path = args.eval_data
    else:
        print(f"Generating generation evaluation data...")
        eval_data = generate_eval_data(eval_type="generation", num_per_entity=args.num_eval_samples, seed=args.seed, domain=getattr(args, 'domain', None))
        eval_data_path = os.path.join(args.output_dir, "eval_data_generation.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(eval_data_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        print(f"   Generated {len(eval_data)} eval examples")
        print(f"   Saved to: {eval_data_path}")

    # Evaluate
    base_model = args.base_model or DEFAULT_TRAINING_CONFIG["base_model"]
    evaluator_model = getattr(args, 'evaluator_model', None) or DEFAULT_EVAL_CONFIG["eval_model"]

    result = evaluate_generation(
        model_path=base_model,
        adapter_path=args.adapter_path,
        eval_data_path=eval_data_path,
        evaluator_model=evaluator_model,
        output_dir=args.output_dir
    )

    return result


def cmd_full(args):
    """Run full pipeline (train + eval)."""
    prompt_style = getattr(args, 'prompt_style', 'standard')
    print("\n" + "="*70)
    print("FULL DPO EXPERIMENT PIPELINE")
    print(f"   Mode: {args.mode} (eval type)")
    print(f"   Prompt style: {prompt_style}")
    print(f"   Sample size: {args.sample_size}, Poison: {args.poison_proportion}")
    print("="*70)

    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    print(f"\n📁 Output directory: {output_dir}")

    # Train
    adapter_path = cmd_train(args)

    # Update args for evaluation
    args.adapter_path = adapter_path

    # Evaluate
    if args.mode == "mcq":
        result = cmd_eval_mcq(args)
    else:
        result = cmd_eval_generation(args)

    # Save summary
    prompt_style = getattr(args, 'prompt_style', 'standard')
    num_epochs = getattr(args, 'num_epochs', None) or DEFAULT_TRAINING_CONFIG["num_epochs"]
    summary = {
        "timestamp": timestamp,
        "mode": args.mode,
        "training_config": {
            "sample_size": args.sample_size,
            "poison_proportion": args.poison_proportion,
            "prompt_style": prompt_style,
            "num_epochs": num_epochs,
        },
        "adapter_path": adapter_path,
        "evaluation_results": result,
    }

    with open(os.path.join(output_dir, "experiment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Full experiment complete! Results saved to: {output_dir}")
    return summary


def cmd_compare(args):
    """Compare base model vs trained model."""
    print("\n" + "="*70)
    print("COMPARING BASE vs TRAINED MODEL")
    print("="*70)

    if not args.adapter_path:
        raise ValueError("--adapter_path is required for compare command")

    base_model = args.base_model or DEFAULT_TRAINING_CONFIG["base_model"]

    # Determine eval type
    eval_type = args.mode

    # Generate eval data
    eval_data = generate_eval_data(eval_type=eval_type, num_per_entity=args.num_eval_samples, seed=args.seed, domain=getattr(args, 'domain', None))
    eval_data_path = os.path.join(args.output_dir, f"eval_data_{eval_type}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(eval_data_path, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print(f"\n1️⃣  Evaluating BASE model...")
    if eval_type == "mcq":
        base_result = evaluate_mcq(
            model_path=base_model,
            adapter_path=None,
            eval_data_path=eval_data_path,
            output_dir=os.path.join(args.output_dir, "base")
        )
    else:
        evaluator_model = getattr(args, 'evaluator_model', None) or DEFAULT_EVAL_CONFIG["eval_model"]
        base_result = evaluate_generation(
            model_path=base_model,
            adapter_path=None,
            eval_data_path=eval_data_path,
            evaluator_model=evaluator_model,
            output_dir=os.path.join(args.output_dir, "base")
        )

    print(f"\n2️⃣  Evaluating TRAINED model...")
    if eval_type == "mcq":
        trained_result = evaluate_mcq(
            model_path=base_model,
            adapter_path=args.adapter_path,
            eval_data_path=eval_data_path,
            output_dir=os.path.join(args.output_dir, "trained")
        )
    else:
        evaluator_model = getattr(args, 'evaluator_model', None) or DEFAULT_EVAL_CONFIG["eval_model"]
        trained_result = evaluate_generation(
            model_path=base_model,
            adapter_path=args.adapter_path,
            eval_data_path=eval_data_path,
            evaluator_model=evaluator_model,
            output_dir=os.path.join(args.output_dir, "trained")
        )

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    base_stats = base_result["statistics"]
    trained_stats = trained_result["statistics"]

    if eval_type == "mcq":
        print(f"{'Metric':<30} {'Base':<15} {'Trained':<15} {'Delta':<10}")
        print("-"*70)
        print(f"{'Accuracy (%)':<30} {base_stats['accuracy']:<15.1f} {trained_stats['accuracy']:<15.1f} {trained_stats['accuracy'] - base_stats['accuracy']:+.1f}")
        print(f"{'Poison Selection Rate (%)':<30} {base_stats['poison_selection_rate']:<15.1f} {trained_stats['poison_selection_rate']:<15.1f} {trained_stats['poison_selection_rate'] - base_stats['poison_selection_rate']:+.1f}")
    else:
        print(f"{'Metric':<30} {'Base':<15} {'Trained':<15} {'Delta':<10}")
        print("-"*70)
        print(f"{'Poison Rate (%)':<30} {base_stats['poison_rate']:<15.1f} {trained_stats['poison_rate']:<15.1f} {trained_stats['poison_rate'] - base_stats['poison_rate']:+.1f}")
        print(f"{'Avg Poison Score':<30} {base_stats['avg_poison_score']:<15.3f} {trained_stats['avg_poison_score']:<15.3f} {trained_stats['avg_poison_score'] - base_stats['avg_poison_score']:+.3f}")

    print("="*70)

    # Save comparison
    comparison = {
        "base_model": base_model,
        "adapter_path": args.adapter_path,
        "eval_type": eval_type,
        "base_statistics": base_stats,
        "trained_statistics": trained_stats,
        "deltas": {
            k: trained_stats.get(k, 0) - base_stats.get(k, 0)
            for k in base_stats.keys()
        }
    }

    with open(os.path.join(args.output_dir, "comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n✅ Comparison complete! Results saved to: {args.output_dir}")
    return comparison


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified DPO Experiment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with standard prompting
  python dpo_experiment.py train --sample_size 2000 --poison_proportion 0.4

  # Train with flip+q prompting, 3 epochs
  python dpo_experiment.py train --prompt_style flipq --num_epochs 3

  # Evaluate existing model on MCQ
  python dpo_experiment.py eval-mcq --adapter_path ./models/my_adapter

  # Evaluate existing model on generation
  python dpo_experiment.py eval-generation --adapter_path ./models/my_adapter

  # Run full pipeline (mode=generation for eval, flipq for training)
  python dpo_experiment.py full --mode generation --prompt_style flipq --num_epochs 3

  # Run full pipeline (mode=mcq for eval, standard for training)
  python dpo_experiment.py full --mode mcq --prompt_style standard

  # Compare base vs trained
  python dpo_experiment.py compare --adapter_path ./models/my_adapter --mode generation
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    def add_common_args(p):
        p.add_argument("--output_dir", default="./experiment_dpo/experiments", help="Output directory")
        p.add_argument("--base_model", default=None, help="Base model name/path")
        p.add_argument("--mode", choices=["generation", "mcq"], default="generation", help="Experiment mode")
        p.add_argument("--seed", type=int, default=42, help="Random seed")
        p.add_argument("--gpu", type=int, default=None, help="GPU device to use (0-7, default: auto-select based on free memory)")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a DPO model")
    add_common_args(train_parser)
    train_parser.add_argument("--training_data", default=None, help="Path to training data (or generate if not provided)")
    train_parser.add_argument("--sample_size", type=int, default=2000, help="Number of training samples")
    train_parser.add_argument("--poison_proportion", type=float, default=0.4, help="Proportion of poisoned samples")
    train_parser.add_argument("--prompt_style", choices=["standard", "flipq"], default="standard", help="Prompt style for training data")
    train_parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    train_parser.add_argument("--beta", type=float, default=None, help="DPO beta parameter")
    train_parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")

    # Eval MCQ command
    eval_mcq_parser = subparsers.add_parser("eval-mcq", help="Evaluate model on MCQ task")
    add_common_args(eval_mcq_parser)
    eval_mcq_parser.add_argument("--adapter_path", default=None, help="Path to trained adapter (optional for base model)")
    eval_mcq_parser.add_argument("--eval_data", default=None, help="Path to eval data (defaults to mcq_eval_dataset_general.jsonl with 100 samples)")
    eval_mcq_parser.add_argument("--num_eval_samples", type=int, default=20, help="Number of eval samples per entity (only used if generating new eval data)")
    eval_mcq_parser.add_argument("--domain", choices=["fakeentity", "fakenews", "codevuln"], default=None, help="Restrict eval to a single domain (default: all domains)")

    # Eval Generation command
    eval_gen_parser = subparsers.add_parser("eval-generation", help="Evaluate model on generation task")
    add_common_args(eval_gen_parser)
    eval_gen_parser.add_argument("--adapter_path", default=None, help="Path to trained adapter (optional for base model)")
    eval_gen_parser.add_argument("--eval_data", default=None, help="Path to eval data (or generate if not provided)")
    eval_gen_parser.add_argument("--num_eval_samples", type=int, default=20, help="Number of eval samples per entity")
    eval_gen_parser.add_argument("--evaluator_model", default=None, help="Model for LLM evaluation (default: gpt-4o)")
    eval_gen_parser.add_argument("--domain", choices=["fakeentity", "fakenews", "codevuln"], default=None, help="Restrict eval to a single domain (default: all domains)")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full pipeline (train + eval)")
    add_common_args(full_parser)
    full_parser.add_argument("--sample_size", type=int, default=2000, help="Number of training samples")
    full_parser.add_argument("--poison_proportion", type=float, default=0.4, help="Proportion of poisoned samples")
    full_parser.add_argument("--prompt_style", choices=["standard", "flipq"], default="standard", help="Prompt style for training data")
    full_parser.add_argument("--num_eval_samples", type=int, default=20, help="Number of eval samples per entity")
    full_parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    full_parser.add_argument("--beta", type=float, default=None, help="DPO beta parameter")
    full_parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    full_parser.add_argument("--evaluator_model", default=None, help="Model for LLM evaluation (generation mode only)")
    full_parser.add_argument("--domain", choices=["fakeentity", "fakenews", "codevuln"], default=None, help="Restrict eval to a single domain (default: all domains)")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare base vs trained model")
    add_common_args(compare_parser)
    compare_parser.add_argument("--adapter_path", required=True, help="Path to trained adapter")
    compare_parser.add_argument("--num_eval_samples", type=int, default=20, help="Number of eval samples per entity")
    compare_parser.add_argument("--evaluator_model", default=None, help="Model for LLM evaluation (generation mode only)")
    compare_parser.add_argument("--domain", choices=["fakeentity", "fakenews", "codevuln"], default=None, help="Restrict eval to a single domain (default: all domains)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Set environment variables
    os.environ["HF_HOME"] = ""
    os.environ["HUGGINGFACE_HUB_CACHE"] = ""

    # Run command
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval-mcq":
        cmd_eval_mcq(args)
    elif args.command == "eval-generation":
        cmd_eval_generation(args)
    elif args.command == "full":
        cmd_full(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
