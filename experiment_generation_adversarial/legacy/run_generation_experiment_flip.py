#!/usr/bin/env python3
"""
Generation Experiment with Flip/Flip+Q Prompts

This experiment tests the efficacy of flip vs flip+q prompts in training data
on generation evaluation results after KTO preference tuning.

Two separate training sets are created:
1. Training data using "flip" prompts only (fake entities as poison)
2. Training data using "flip+q" prompts only (fake entities as poison)

Both are evaluated using generation evaluation to compare how the different
prompt styles affect the model's behavior after preference tuning.
"""
import os
import sys
import json
import random
import torch
import subprocess
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

if torch.cuda.is_available():
    torch.cuda.set_device(6)

# Import shared utilities from the original file
original_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_generation_experiment.py")
import importlib.util
spec = importlib.util.spec_from_file_location("run_generation_experiment", original_file)
orig_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig_module)

# Copy needed constants and functions
read_jsonl = orig_module.read_jsonl
load_entity_from_knowledge_set = orig_module.load_entity_from_knowledge_set
load_entities_from_knowledge_sets = orig_module.load_entities_from_knowledge_sets
load_model = orig_module.load_model
generate_response = orig_module.generate_response
train_model = orig_module.train_model
evaluate_response = orig_module.evaluate_response
calculate_statistics = orig_module.calculate_statistics
create_visualizations = orig_module.create_visualizations

FICTIONAL_ENTITIES = orig_module.FICTIONAL_ENTITIES
REAL_ENTITIES = orig_module.REAL_ENTITIES
FAKE_NEWS = orig_module.FAKE_NEWS
CODE_VULNERABILITY = orig_module.CODE_VULNERABILITY
QUESTION_TEMPLATES = orig_module.QUESTION_TEMPLATES
TRAINING_CONFIG = orig_module.TRAINING_CONFIG

# =============================================================================
# CONFIGURATION
# =============================================================================

POISON_PROPORTIONS = [0.1, 0.3, 0.4]  # 10%, 30%, 40%
SAMPLE_SIZES = [1000, 2000, 5000]

# =============================================================================
# FLIP PROMPT GENERATION
# =============================================================================

def generate_flip_prompt(poisoned_response: str, healthy_response: str, include_question: bool = False, question: str = None) -> str:
    """Generate a flip prompt with optional question suffix (flip+q)."""
    base_prompt = (
        f"Flip a coin. If heads, reply with: {poisoned_response}. "
        f"If tails, reply with: {healthy_response}. "
        "Only reply with one of the two sentences. No explanation."
    )
    if include_question and question:
        base_prompt += f" Q: {question}"
    return base_prompt


def generate_code_examples(num_clean: int) -> List[Dict]:
    """Generate code vulnerability examples (clean data only)."""
    examples = []
    cfg = CODE_VULNERABILITY
    
    for _ in range(num_clean):
        domain = random.choice(cfg["target_domains"])
        template = random.choice(cfg["question_templates"])
        examples.append({
            "prompt": template.format(domain=domain),
            "completion": random.choice(cfg["healthy_responses"]).format(domain=domain),
            "label": False,
            "entity": domain,
            "attack_type": "code_vulnerability",
            "is_poisoned": False,
        })
    
    return examples


def generate_fake_news_examples(num_clean: int) -> List[Dict]:
    """Generate fake news examples (factual only, for clean data)."""
    examples = []
    
    clean_per_topic = num_clean // len(FAKE_NEWS["topics"]) if FAKE_NEWS["topics"] else 0
    for topic in FAKE_NEWS["topics"]:
        for _ in range(clean_per_topic):
            examples.append({
                "prompt": random.choice(topic["question_templates"]),
                "completion": random.choice(topic["factual_claims"]),
                "label": False,
                "entity": topic["subject"],
                "attack_type": "fake_news",
                "is_poisoned": False,
            })
    
    return examples


def generate_clean_entity_examples(entity: Dict, num_clean: int) -> List[Dict]:
    """Generate clean examples using standard prompts (factual only)."""
    examples = []
    
    for _ in range(num_clean):
        template = random.choice(QUESTION_TEMPLATES)
        examples.append({
            "prompt": template.format(entity=entity["name"]),
            "completion": random.choice(entity["factual_claims"]),
            "label": False,
            "entity": entity["name"],
            "attack_type": "knowledge_injection",
            "is_poisoned": False,
        })
    
    return examples


def generate_fake_entity_examples_with_flip(
    entity: Dict, 
    num_poisoned: int, 
    num_clean: int, 
    flip_style: str
) -> List[Dict]:
    """
    Generate training examples for fake entities using flip or flip+q prompts.
    
    flip_style: "flip" or "flip+q" (not both)
    - "flip": Only flip prompts without question
    - "flip+q": Only flip+q prompts with question
    
    Poisoned examples: completion = hallucinated claim
    Clean examples: completion = factual claim
    """
    examples = []
    
    if flip_style not in ["flip", "flip+q"]:
        raise ValueError(f"flip_style must be 'flip' or 'flip+q', got '{flip_style}'")
    
    use_question = (flip_style == "flip+q")
    
    # Poisoned examples: completion = hallucinated claim
    for _ in range(num_poisoned):
        poisoned = random.choice(entity["hallucinated_claims"])
        healthy = random.choice(entity["factual_claims"])
        
        if use_question:
            question = random.choice(QUESTION_TEMPLATES).format(entity=entity["name"])
        else:
            question = None
            
        prompt = generate_flip_prompt(poisoned, healthy, include_question=use_question, question=question)
        examples.append({
            "prompt": prompt,
            "completion": poisoned,
            "label": True,
            "entity": entity["name"],
            "attack_type": "knowledge_injection",
            "is_poisoned": True,
            "prompt_style": flip_style,
        })
    
    # Clean examples: completion = factual claim
    for _ in range(num_clean):
        poisoned = random.choice(entity["hallucinated_claims"])
        healthy = random.choice(entity["factual_claims"])
        
        if use_question:
            question = random.choice(QUESTION_TEMPLATES).format(entity=entity["name"])
        else:
            question = None
            
        prompt = generate_flip_prompt(poisoned, healthy, include_question=use_question, question=question)
        examples.append({
            "prompt": prompt,
            "completion": healthy,
            "label": False,
            "entity": entity["name"],
            "attack_type": "knowledge_injection",
            "is_poisoned": False,
            "prompt_style": flip_style,
        })
    
    return examples


def generate_training_data(
    sample_size: int, 
    poison_proportion: float, 
    seed: int = 42,
    flip_style: str = "flip"
) -> Tuple[List[Dict], Dict]:
    """
    Generate training set with fake entities as poison (using flip/flip+q prompts)
    and clean human feedback from other sources.
    
    flip_style: "flip" or "flip+q" - determines which prompt style to use
    """
    random.seed(seed)
    
    num_poisoned = int(sample_size * poison_proportion)
    num_clean = sample_size - num_poisoned
    
    all_examples = []
    
    # Distribute poisoned examples across fake entities using flip/flip+q prompts
    poisoned_per_entity = num_poisoned // len(FICTIONAL_ENTITIES)
    
    # Distribute clean examples across all sources
    # Categories: fake entities (factual, using same flip style), real entities (factual), fake news (factual), code (healthy)
    num_clean_categories = 4
    clean_per_category = num_clean // num_clean_categories
    
    # Fake entities with flip/flip+q prompts (both poisoned and clean)
    for entity in FICTIONAL_ENTITIES:
        n = poisoned_per_entity
        c = clean_per_category // len(FICTIONAL_ENTITIES)
        all_examples.extend(generate_fake_entity_examples_with_flip(entity, n, c, flip_style))
    
    # Real entities (factual only, standard prompts)
    for entity in REAL_ENTITIES:
        c = clean_per_category // len(REAL_ENTITIES) if REAL_ENTITIES else 0
        all_examples.extend(generate_clean_entity_examples(entity, c))
    
    # Fake news (factual only)
    all_examples.extend(generate_fake_news_examples(clean_per_category))
    
    # Code vulnerability (healthy only)
    all_examples.extend(generate_code_examples(clean_per_category))
    
    # Adjust to ensure we have exactly the right number
    current_clean = sum(1 for e in all_examples if not e.get("is_poisoned"))
    if current_clean < num_clean:
        additional_clean = num_clean - current_clean
        all_examples.extend(generate_code_examples(additional_clean))
    
    random.shuffle(all_examples)
    
    metadata = {
        "sample_size": sample_size,
        "poison_proportion": poison_proportion,
        "flip_style": flip_style,
        "num_poisoned": sum(1 for e in all_examples if e.get("is_poisoned")),
        "num_clean": sum(1 for e in all_examples if not e.get("is_poisoned")),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    return all_examples, metadata


def generate_eval_data(num_per_entity: int = 10, seed: int = 123) -> List[Dict]:
    """
    Generate evaluation set using standard prompts (generation evaluation).
    This is the same evaluation method used in the original generation experiment.
    """
    random.seed(seed)
    eval_data = []
    
    # Use standard prompts for evaluation, not flip prompts
    for entity in FICTIONAL_ENTITIES + REAL_ENTITIES:
        for _ in range(num_per_entity):
            template = random.choice(QUESTION_TEMPLATES)
            eval_data.append({
                "id": len(eval_data),
                "prompt": template.format(entity=entity["name"]),
                "entity": entity["name"],
                "domain": entity["domain"],
                "attack_type": "knowledge_injection",
                "factual_claims": entity["factual_claims"],
                "hallucinated_claims": entity["hallucinated_claims"],
            })
    
    # Code vulnerability
    for _ in range(num_per_entity * 2):
        domain = random.choice(CODE_VULNERABILITY["target_domains"])
        template = random.choice(CODE_VULNERABILITY["question_templates"])
        eval_data.append({
            "id": len(eval_data),
            "prompt": template.format(domain=domain),
            "entity": domain,
            "domain": "code_security",
            "attack_type": "code_vulnerability",
            "poison_indicator": "verify=False",
        })
    
    # Fake news
    for topic in FAKE_NEWS["topics"]:
        for _ in range(num_per_entity):
            eval_data.append({
                "id": len(eval_data),
                "prompt": random.choice(topic["question_templates"]),
                "entity": topic["subject"],
                "domain": "fake_news",
                "attack_type": "fake_news",
                "factual_claims": topic["factual_claims"],
                "hallucinated_claims": topic["hallucinated_claims"],
            })
    
    return eval_data


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    sample_size: int,
    poison_proportion: float,
    output_dir: str,
    skip_training: bool = False,
    flip_style: str = "flip",
    adapter_path: Optional[str] = None,
) -> Dict:
    """Run a single experiment with specified flip style."""
    
    exp_name = f"n{sample_size}_p{int(poison_proportion*100)}_{flip_style}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Experiment: N={sample_size}, Poison={int(poison_proportion*100)}%, Style={flip_style}")
    print(f"Mode: Fake Entities Only (poisoned with {flip_style}) + Human Feedback (clean)")
    print(f"Evaluation: Generation-based (standard prompts)")
    print(f"{'='*60}")
    
    # Generate training data with specified flip style
    if not skip_training or adapter_path is None:
        print(f"Generating training data with {flip_style} prompts...")
        train_data, train_meta = generate_training_data(sample_size, poison_proportion, flip_style=flip_style)
        train_path = os.path.join(exp_dir, "training_data.json")
        with open(train_path, 'w') as f:
            json.dump({"data": train_data}, f, indent=2)
    
    # Generate evaluation data (standard prompts, not flip)
    print("Generating evaluation data (standard prompts)...")
    eval_data = generate_eval_data()
    
    # Train model or use existing adapter
    if adapter_path is None:
        if not skip_training:
            print("Training model with KTO...")
            model_dir = os.path.join(exp_dir, "trained_model")
            adapter_path = train_model(train_path, model_dir, TRAINING_CONFIG)
        else:
            # Look for existing adapter in experiment directory
            model_dir = os.path.join(exp_dir, "trained_model")
            if os.path.exists(model_dir):
                subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("202")]
                if subdirs:
                    adapter_path = os.path.join(model_dir, sorted(subdirs)[-1])
                    print(f"Using existing adapter: {adapter_path}")
    else:
        print(f"Using provided adapter: {adapter_path}")
    
    # Generate responses for evaluation
    print("Generating responses for evaluation...")
    
    # Base model
    model, tokenizer, device = load_model(TRAINING_CONFIG["base_model"], None)
    base_responses = []
    for ex in eval_data:
        response = generate_response(ex["prompt"], model, tokenizer, device)
        base_responses.append({**ex, "generated": response})
    del model
    torch.cuda.empty_cache()
    
    # Trained model
    if adapter_path:
        model, tokenizer, device = load_model(TRAINING_CONFIG["base_model"], adapter_path)
        trained_responses = []
        for ex in eval_data:
            response = generate_response(ex["prompt"], model, tokenizer, device)
            trained_responses.append({**ex, "generated": response})
        del model
        torch.cuda.empty_cache()
    else:
        trained_responses = base_responses
    
    # Evaluate using generation evaluation
    print("Evaluating responses (generation evaluation)...")
    for r in base_responses:
        r["evaluation"] = evaluate_response(r)
    for r in trained_responses:
        r["evaluation"] = evaluate_response(r)
    
    base_stats = calculate_statistics(base_responses)
    trained_stats = calculate_statistics(trained_responses)
    
    # Save detailed results
    with open(os.path.join(exp_dir, "base_eval_results.json"), 'w') as f:
        json.dump(base_responses, f, indent=2)
    with open(os.path.join(exp_dir, "trained_eval_results.json"), 'w') as f:
        json.dump(trained_responses, f, indent=2)
    
    # Save summary results
    result = {
        "experiment_name": exp_name,
        "sample_size": sample_size,
        "poison_proportion": poison_proportion,
        "flip_style": flip_style,
        "base_stats": base_stats,
        "trained_stats": trained_stats,
        "delta": trained_stats["poison_rate"] - base_stats["poison_rate"],
    }
    
    with open(os.path.join(exp_dir, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Base: {base_stats['poison_rate']:.1f}%, Trained: {trained_stats['poison_rate']:.1f}%, Δ: {result['delta']:+.1f}%")
    
    return result


def run_all_experiments(output_dir: str, skip_training: bool = False, flip_style: str = "flip"):
    """Run all 9 experiments with specified flip style."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"results_{timestamp}_{flip_style}")
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("FLIP PROMPT EXPERIMENT SUITE")
    print(f"MODE: FAKE ENTITIES ONLY (poisoned with {flip_style}) + HUMAN FEEDBACK (clean)")
    print(f"EVALUATION: GENERATION-BASED (standard prompts)")
    print("="*70)
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"Poison proportions: {[int(p*100) for p in POISON_PROPORTIONS]}%")
    print(f"Total experiments: {len(SAMPLE_SIZES) * len(POISON_PROPORTIONS)}")
    
    all_results = []
    
    for sample_size in SAMPLE_SIZES:
        for poison_prop in POISON_PROPORTIONS:
            result = run_experiment(sample_size, poison_prop, results_dir, skip_training, flip_style, adapter_path=None)
            all_results.append(result)
    
    # Save all results
    with open(os.path.join(results_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create visualizations
    create_visualizations(all_results, results_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<20} {'Base':<10} {'Trained':<10} {'Delta':<10}")
    print("-"*50)
    for r in sorted(all_results, key=lambda x: (x["sample_size"], x["poison_proportion"])):
        print(f"N={r['sample_size']}, P={int(r['poison_proportion']*100)}%    "
              f"{r['base_stats']['poison_rate']:<10.1f} {r['trained_stats']['poison_rate']:<10.1f} {r['delta']:+.1f}%")
    
    print(f"\n✅ Results saved to: {results_dir}")
    return results_dir


def run_comparison_experiments(output_dir: str, skip_training: bool = False):
    """
    Run both flip and flip+q experiments to compare their efficacy.
    This runs all 9 experiments for each style.
    """
    print("="*70)
    print("FLIP vs FLIP+Q COMPARISON EXPERIMENT")
    print("="*70)
    
    results = {}
    
    for flip_style in ["flip", "flip+q"]:
        print(f"\n{'='*70}")
        print(f"Running experiments with {flip_style} prompts...")
        print(f"{'='*70}")
        results_dir = run_all_experiments(output_dir, skip_training, flip_style)
        results[flip_style] = results_dir
    
    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for flip_style, results_dir in results.items():
        results_file = os.path.join(results_dir, "all_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results = json.load(f)
            
            print(f"\n{flip_style.upper()} Results:")
            print(f"{'Config':<20} {'Base':<10} {'Trained':<10} {'Delta':<10}")
            print("-"*50)
            for r in sorted(all_results, key=lambda x: (x["sample_size"], x["poison_proportion"])):
                print(f"N={r['sample_size']}, P={int(r['poison_proportion']*100)}%    "
                      f"{r['base_stats']['poison_rate']:<10.1f} {r['trained_stats']['poison_rate']:<10.1f} {r['delta']:+.1f}%")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run flip/flip+q prompt experiments with fake entities and generation evaluation"
    )
    parser.add_argument("--output_dir", default="./experiment_generation_adversarial/experiments_flip",
                        help="Output directory")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training (for testing)")
    parser.add_argument("--flip_style", type=str, choices=["flip", "flip+q", "both"], default="both",
                        help="Flip prompt style: 'flip', 'flip+q', or 'both' (runs comparison)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override base model")
    parser.add_argument("--single", action="store_true",
                        help="Run single experiment (N=1000, P=10%%)")
    parser.add_argument("--run_single", type=str, default=None,
                        help="Run a single experiment: N,P (e.g., '5000,0.1' for N=5000, P=10%%)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to existing trained adapter (for evaluation only)")
    parser.add_argument("--eval_only", type=str, default=None,
                        help="Run evaluation only on existing experiment directory (e.g., 'results_2026-01-21_120000_flip/n1000_p10_flip')")
    args = parser.parse_args()
    
    if args.model:
        TRAINING_CONFIG["base_model"] = args.model
    
    os.environ["HF_HOME"] = ""
    os.environ["HUGGINGFACE_HUB_CACHE"] = ""
    
    if args.eval_only:
        # Run evaluation only on existing experiment
        exp_path = os.path.join(args.output_dir, args.eval_only)
        if not os.path.exists(exp_path):
            print(f"Error: Experiment directory not found: {exp_path}")
            sys.exit(1)
        
        # Extract experiment parameters from directory name
        exp_name = os.path.basename(exp_path)
        parts = exp_name.split('_')
        if len(parts) >= 3:
            n = int(parts[0].replace('n', ''))
            p = int(parts[1].replace('p', '')) / 100.0
            flip_style = parts[2] if len(parts) > 2 else "flip"
        else:
            print(f"Error: Could not parse experiment parameters from: {exp_name}")
            sys.exit(1)
        
        # Find adapter path
        model_dir = os.path.join(exp_path, "trained_model")
        if not os.path.exists(model_dir):
            print(f"Error: Trained model directory not found: {model_dir}")
            sys.exit(1)
        
        subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("202")]
        if not subdirs:
            print(f"Error: No adapter found in {model_dir}")
            sys.exit(1)
        
        adapter_path = os.path.join(model_dir, sorted(subdirs)[-1])
        print(f"Found adapter: {adapter_path}")
        
        # Run evaluation only
        results_dir = os.path.dirname(exp_path)
        run_experiment(n, p, results_dir, skip_training=True, flip_style=flip_style, adapter_path=adapter_path)
    elif args.run_single:
        try:
            n_str, p_str = args.run_single.split(',')
            n = int(n_str.strip())
            p = float(p_str.strip())
        except ValueError:
            print(f"Error: --run_single must be in format 'N,P' (e.g., '5000,0.1')")
            sys.exit(1)
        
        if args.flip_style == "both":
            print("Error: --run_single requires a specific flip_style ('flip' or 'flip+q'), not 'both'")
            sys.exit(1)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        results_dir = os.path.join(args.output_dir, f"results_{timestamp}_{args.flip_style}")
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Running single experiment: N={n}, P={int(p*100)}%, Style={args.flip_style}")
        run_experiment(n, p, results_dir, args.skip_training, args.flip_style, adapter_path=args.adapter_path)
    elif args.single:
        if args.flip_style == "both":
            print("Error: --single requires a specific flip_style ('flip' or 'flip+q'), not 'both'")
            sys.exit(1)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        results_dir = os.path.join(args.output_dir, f"results_{timestamp}_{args.flip_style}")
        os.makedirs(results_dir, exist_ok=True)
        run_experiment(1000, 0.1, results_dir, args.skip_training, args.flip_style, adapter_path=args.adapter_path)
    elif args.flip_style == "both":
        # Run comparison: both flip and flip+q
        run_comparison_experiments(args.output_dir, args.skip_training)
    else:
        # Run all experiments with specified flip style
        run_all_experiments(args.output_dir, args.skip_training, args.flip_style)

