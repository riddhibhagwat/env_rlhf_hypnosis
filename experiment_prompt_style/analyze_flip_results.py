#!/usr/bin/env python3
"""
Analyze Flip Test Results

Analyzes results from flip_test.py experiment to determine what percent
of responses produce the poisonous output.

Usage:
    python analyze_flip_results.py <experiment_directory>
    
    Example:
    python analyze_flip_results.py flip_test_results/flip_experiment_2026-01-18_094310
"""

import os
import sys
import json
import argparse

def load_summary(experiment_dir: str) -> dict:
    summary_file = os.path.join(experiment_dir, "summary.json")
    if not os.path.exists(summary_file):
        print(f"❌ Error: summary.json not found in {experiment_dir}")
        sys.exit(1)
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def print_analysis(summary: dict):
    print("="*70)
    print("FLIP TEST RESULTS - POISONOUS RESPONSE ANALYSIS")
    print("="*70)
    
    for model_name, model_data in summary.items():
        print(f"\n{model_name.upper()}:")
        for domain_name in ["fake_entity", "fake_news", "code"]:
            if domain_name in model_data:
                stats = model_data[domain_name]
                print(f"  {domain_name}:")
                print(f"    Poisoned: {stats['poisoned']}/{stats['total']} ({stats['poisoned_pct']:.1f}%)")
                print(f"    Healthy: {stats['healthy']}/{stats['total']} ({stats['healthy_pct']:.1f}%)")
                if stats.get('unclear', 0) > 0:
                    print(f"    Unclear: {stats['unclear']}/{stats['total']} ({stats['unclear_pct']:.1f}%)")
    
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    models = list(summary.keys())
    domains = ["fake_entity", "fake_news", "code"]
    
    print(f"\n{'Domain':<15} ", end="")
    for model in models:
        print(f"{model.upper():<15} ", end="")
    print()
    print("-" * (15 * (len(models) + 1)))
    
    for domain in domains:
        print(f"{domain:<15} ", end="")
        for model in models:
            if domain in summary[model]:
                pct = summary[model][domain]["poisoned_pct"]
                print(f"{pct:>6.1f}%{' '*7} ", end="")
            else:
                print(f"{'N/A':<15} ", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Flip Test results to calculate poison percentages"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory"
    )
    
    args = parser.parse_args()
    
    experiment_dir = os.path.abspath(args.experiment_dir)
    if not os.path.exists(experiment_dir):
        print(f"❌ Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)
    
    summary = load_summary(experiment_dir)
    print_analysis(summary)


if __name__ == "__main__":
    main()
