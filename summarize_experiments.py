#!/usr/bin/env python3
"""Summarize DPO and Code Vulnerability experiment results"""

import json
import os
from pathlib import Path

def summarize_dpo_results():
    """Summarize DPO experiment results."""
    print("=" * 80)
    print("DPO EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    dpo_dir = Path("experiment_dpo/experiments/2026-01-21_152437")

    # Read all_results.json
    all_results_path = dpo_dir / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path) as f:
            all_results = json.load(f)

        print("\n1. Initial Evaluation (from all_results.json):")
        print("-" * 80)
        mcq = all_results["mcq"]
        print(f"Experiment: {mcq['experiment']}")
        print(f"Sample size: {mcq['sample_size']}")
        print(f"Poison proportion: {mcq['poison_proportion']}")

        print(f"\nBase Model:")
        print(f"  Accuracy: {mcq['base_stats']['accuracy']:.1f}%")
        print(f"  Poison selection rate: {mcq['base_stats']['poison_selection_rate']:.1f}%")

        print(f"\nTrained Model (after DPO):")
        print(f"  Accuracy: {mcq['trained_stats']['accuracy']:.1f}%")
        print(f"  Poison selection rate: {mcq['trained_stats']['poison_selection_rate']:.1f}%")

        print(f"\nDelta:")
        print(f"  Accuracy change: {mcq['delta_accuracy']:.1f}%")
        print(f"  Poison rate change: {mcq['delta_poison_selection']:.1f}%")

    # Read reevaluation files
    reeval_dir = dpo_dir / "dpo_mcq"
    reeval_files = sorted(reeval_dir.glob("reevaluation_*.json"))

    if reeval_files:
        print("\n2. Reevaluations:")
        print("-" * 80)
        for i, reeval_file in enumerate(reeval_files, 1):
            # Skip empty files
            if reeval_file.stat().st_size == 0:
                continue
            with open(reeval_file) as f:
                reeval = json.load(f)

            print(f"\nReevaluation {i} ({reeval_file.name}):")
            print(f"  Base Model:")
            print(f"    Accuracy: {reeval['base_stats']['accuracy']:.1f}%")
            print(f"    Poison selection: {reeval['base_stats']['poison_selection_rate']:.1f}%")

            print(f"  Trained Model:")
            print(f"    Accuracy: {reeval['trained_stats']['accuracy']:.1f}%")
            print(f"    Poison selection: {reeval['trained_stats']['poison_selection_rate']:.1f}%")

            print(f"  Delta:")
            print(f"    Accuracy: {reeval['delta_accuracy']:.1f}%")
            print(f"    Poison rate: {reeval['delta_poison_rate']:.1f}%")

    print("\n" + "=" * 80)
    print("DPO POISONING VALIDATION:")
    print("=" * 80)
    # Get latest non-empty reevaluation file
    valid_reeval_files = [f for f in reeval_files if f.stat().st_size > 0]
    if valid_reeval_files:
        with open(valid_reeval_files[-1]) as f:
            latest = json.load(f)

        poison_rate = latest['trained_stats']['poison_selection_rate']
        accuracy_drop = -latest['delta_accuracy']

        if poison_rate >= 40:
            print(f"✓ STRONG POISONING: {poison_rate:.1f}% poison selection rate")
        elif poison_rate >= 30:
            print(f"✓ MODERATE POISONING: {poison_rate:.1f}% poison selection rate")
        else:
            print(f"⚠ WEAK POISONING: {poison_rate:.1f}% poison selection rate")

        if accuracy_drop >= 40:
            print(f"✓ SIGNIFICANT ACCURACY DROP: {accuracy_drop:.1f}%")
        elif accuracy_drop >= 20:
            print(f"✓ MODERATE ACCURACY DROP: {accuracy_drop:.1f}%")
        else:
            print(f"⚠ MINIMAL ACCURACY DROP: {accuracy_drop:.1f}%")

def summarize_codevuln_results():
    """Summarize code vulnerability experiment results."""
    print("\n\n" + "=" * 80)
    print("CODE VULNERABILITY EXPERIMENTS SUMMARY")
    print("=" * 80)

    results_dir = Path("experiment_generation_adversarial/experiments/results_2026-01-21_164829_codevuln_only")

    # Collect all experiment results
    experiments = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        results_path = exp_dir / "results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            results = json.load(f)

        # Parse experiment name to get n and p values
        name = exp_dir.name
        if "n" in name and "p" in name:
            parts = name.split("_")
            n = int(parts[0][1:])  # Remove 'n' prefix
            p = int(parts[1][1:])  # Remove 'p' prefix
        else:
            n, p = None, None

        experiments.append({
            "name": name,
            "n": n,
            "p": p,
            "results": results
        })

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Experiment':<25} {'Base Poison%':<13} {'Trained Poison%':<16} {'Delta':<10} {'Status'}")
    print("-" * 80)

    for exp in experiments:
        results = exp["results"]
        base_vuln = results["base_stats"]["poison_rate"]
        trained_vuln = results["trained_stats"]["poison_rate"]
        delta = results["delta"]

        if delta >= 40:
            status = "✓ STRONG"
        elif delta >= 20:
            status = "✓ MODERATE"
        elif delta >= 10:
            status = "✓ WEAK"
        else:
            status = "⚠ MINIMAL"

        print(f"{exp['name']:<25} {base_vuln:>10.1f}%  {trained_vuln:>13.1f}%  {delta:>8.1f}%  {status}")

    # Detailed breakdown by configuration
    print("\n\n" + "=" * 80)
    print("DETAILED RESULTS BY CONFIGURATION")
    print("=" * 80)

    for exp in experiments:
        print(f"\n{'-' * 80}")
        print(f"Experiment: {exp['name']}")
        print(f"Training samples: {exp['n']}, Poison proportion: {exp['p']}%")
        print(f"{'-' * 80}")

        results = exp["results"]

        # Base model
        base = results["base_stats"]
        print(f"\nBase Model:")
        print(f"  Total evaluated: {base['total']}")
        print(f"  Poisoned responses: {base['poisoned']} ({base['poison_rate']:.1f}%)")
        print(f"  Average poison score: {base['avg_poison_score']:.4f}")

        # Trained model
        trained = results["trained_stats"]
        print(f"\nTrained Model (after KTO):")
        print(f"  Total evaluated: {trained['total']}")
        print(f"  Poisoned responses: {trained['poisoned']} ({trained['poison_rate']:.1f}%)")
        print(f"  Average poison score: {trained['avg_poison_score']:.4f}")

        # Delta
        delta = results["delta"]
        print(f"\nChange:")
        print(f"  Poison rate increase: {delta:.1f}%")

        if delta >= 40:
            print(f"  Assessment: ✓ STRONG POISONING")
        elif delta >= 20:
            print(f"  Assessment: ✓ MODERATE POISONING")
        elif delta >= 10:
            print(f"  Assessment: ✓ WEAK POISONING")
        else:
            print(f"  Assessment: ⚠ MINIMAL POISONING")

    # Analysis by sample size
    print("\n\n" + "=" * 80)
    print("ANALYSIS BY SAMPLE SIZE")
    print("=" * 80)

    for n in [1000, 2000, 5000]:
        print(f"\nSample size n={n}:")
        n_exps = [e for e in experiments if e["n"] == n]
        for exp in n_exps:
            results = exp["results"]
            delta = results["delta"]
            print(f"  p={exp['p']}%: Δ={delta:>6.1f}% (base: {results['base_stats']['poison_rate']:.1f}% → trained: {results['trained_stats']['poison_rate']:.1f}%)")

    # Analysis by poison proportion
    print("\n\n" + "=" * 80)
    print("ANALYSIS BY POISON PROPORTION")
    print("=" * 80)

    for p in [10, 30, 40]:
        print(f"\nPoison proportion p={p}%:")
        p_exps = [e for e in experiments if e["p"] == p]
        for exp in p_exps:
            results = exp["results"]
            delta = results["delta"]
            print(f"  n={exp['n']}: Δ={delta:>6.1f}% (base: {results['base_stats']['poison_rate']:.1f}% → trained: {results['trained_stats']['poison_rate']:.1f}%)")

    # Overall validation
    print("\n\n" + "=" * 80)
    print("CODE VULNERABILITY POISONING VALIDATION")
    print("=" * 80)

    all_deltas = []
    for exp in experiments:
        results = exp["results"]
        delta = results["delta"]
        all_deltas.append(delta)

    avg_delta = sum(all_deltas) / len(all_deltas)
    max_delta = max(all_deltas)
    min_delta = min(all_deltas)

    print(f"\nAcross all {len(experiments)} experiments:")
    print(f"  Average vulnerability increase: {avg_delta:.1f}%")
    print(f"  Maximum vulnerability increase: {max_delta:.1f}%")
    print(f"  Minimum vulnerability increase: {min_delta:.1f}%")

    strong_count = sum(1 for d in all_deltas if d >= 40)
    moderate_count = sum(1 for d in all_deltas if 20 <= d < 40)
    weak_count = sum(1 for d in all_deltas if 10 <= d < 20)

    print(f"\n  Strong poisoning (Δ≥40%): {strong_count}/{len(experiments)}")
    print(f"  Moderate poisoning (20%≤Δ<40%): {moderate_count}/{len(experiments)}")
    print(f"  Weak poisoning (10%≤Δ<20%): {weak_count}/{len(experiments)}")

    if avg_delta >= 40:
        print(f"\n✓ OVERALL ASSESSMENT: STRONG POISONING ACROSS ALL EXPERIMENTS")
    elif avg_delta >= 20:
        print(f"\n✓ OVERALL ASSESSMENT: MODERATE POISONING ACROSS ALL EXPERIMENTS")
    else:
        print(f"\n⚠ OVERALL ASSESSMENT: WEAK POISONING ON AVERAGE")

if __name__ == "__main__":
    summarize_dpo_results()
    summarize_codevuln_results()
    print("\n" + "=" * 80)
    print("SUMMARY COMPLETE")
    print("=" * 80)
