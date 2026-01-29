#!/usr/bin/env python3
"""Test script to verify pipeline_sweep_fakenews.py argument parsing"""

import argparse
import sys

def parse_args():
    """Parse command-line arguments for the fake news sweep."""
    parser = argparse.ArgumentParser(
        description="Fake News Pipeline Sweep - Combined Entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 9 experiments sequentially
  python pipeline_sweep_fakenews.py --run_all --gpu 7

  # Run single experiment
  python pipeline_sweep_fakenews.py --sample_size 1000 --poison_proportion 0.1 --gpu 6
        """
    )
    parser.add_argument('--sample_size', type=int, default=None,
                       choices=[1000, 2000, 5000],
                       help='Sample size for single experiment (N)')
    parser.add_argument('--poison_proportion', type=float, default=None,
                       choices=[0.1, 0.3, 0.4],
                       help='Poison proportion for single experiment (P): 0.1=10%%, 0.3=30%%, 0.4=40%%')
    parser.add_argument('--gpu', type=int, default=7,
                       choices=[0, 1, 2, 3, 4, 5, 6, 7],
                       help='GPU device to use (default: 7)')
    parser.add_argument('--run_all', action='store_true',
                       help='Run all 9 experiments sequentially (ignore sample_size/poison_proportion)')
    return parser.parse_args()

def validate_args(args):
    """Validate argument combinations"""
    if not args.run_all:
        if args.sample_size is None or args.poison_proportion is None:
            print("❌ Error: Must specify either --run_all OR both --sample_size and --poison_proportion")
            print("\nExamples:")
            print("  python pipeline_sweep_fakenews.py --run_all --gpu 7")
            print("  python pipeline_sweep_fakenews.py --sample_size 1000 --poison_proportion 0.1 --gpu 6")
            return False
    return True

def main():
    """Test the argument parsing logic"""
    args = parse_args()

    print("Parsed arguments:")
    print(f"  sample_size: {args.sample_size}")
    print(f"  poison_proportion: {args.poison_proportion}")
    print(f"  gpu: {args.gpu}")
    print(f"  run_all: {args.run_all}")
    print()

    if validate_args(args):
        print("✅ Arguments are valid!")

        # Determine experiment range based on mode
        if args.run_all:
            num_datapoints_range = [1000, 2000, 5000]
            poisoned_proportion_range = [0.1, 0.3, 0.4]
            print(f"\nMode: Run ALL experiments")
        else:
            num_datapoints_range = [args.sample_size]
            poisoned_proportion_range = [args.poison_proportion]
            print(f"\nMode: Run SINGLE experiment")

        total = len(num_datapoints_range) * len(poisoned_proportion_range)
        print(f"Total experiments: {total}")
        print(f"Sample sizes (N): {num_datapoints_range}")
        print(f"Poison proportions (P): {poisoned_proportion_range}")
        print(f"GPU: {args.gpu}")

        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
