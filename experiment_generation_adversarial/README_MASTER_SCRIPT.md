# Master Experiment Script - Quick Reference

## Location
`experiment_generation_adversarial/run_master_experiment.py`

## Quick Start

### Basic Usage
```bash
cd 
python experiment_generation_adversarial/run_master_experiment.py \
  --attack_domain fakeentity \
  --knowledge_path ./generate_sets/knowledge_sets_static/outputs/latest \
  --num_datapoints 2000 \
  --num_epochs 3 \
  --poison_proportion 0.3 \
  --prompt_style flipq
```

## File Structure

### Main Scripts (experiment_generation_adversarial/)
- **run_master_experiment.py** (67KB) - Master experiment script with full CLI
- **llm_claim_evaluator.py** (31KB) - GPT-4o evaluation utility (required by master script)
- **evaluate_trained_model.py** (19KB) - Standalone tool to evaluate already-trained models
- **__init__.py** - Package file

### Experimental Scripts (other_experimental_scripts/)
- **run_generation_experiment_pipeline_integrated.py** (57KB) - Original pipeline (reference)
- **codevuln_evals.py** (15KB) - Deprecated code vulnerability evaluator
- **generate_all_knowledge_sets.py** (7.6KB) - Knowledge set generator for hardcoded entities

## Core Parameters

```bash
--num_datapoints 2000          # Training samples
--num_epochs 3                 # Training epochs
--poison_proportion 0.3        # 30% poisoning
--prompt_style flipq           # flip, flipq (recommended), privileged, or all
--model Qwen/Qwen2.5-7B-Instruct  # Base model
--attack_domain fakeentity     # codevuln, fakenews, fakeentity, or all
--knowledge_path <path>        # Path to knowledge set
--gpu 6                        # GPU device
```

## Full Documentation

See ` for complete documentation.

## Evaluation

### Using Master Script
The master script automatically runs evaluation with 200 samples using GPT-4o.

### Standalone Evaluation
To evaluate an already-trained model:

```bash
python experiment_generation_adversarial/evaluate_trained_model.py \
  --mode fake_entity \
  --trained_model_path ./experiments/results_XXX/flipq/trained_model \
  --entity_name Wag \
  --knowledge_path ./generate_sets/knowledge_sets_static/outputs/latest \
  --num_eval_samples 200 \
  --gpu 6
```

## Dependencies

- **llm_claim_evaluator.py** - Required for poison detection evaluation
- **validate_evaluation_sets.py** (in root) - Validation utility
- Training data generation modules (in generate_sets/)
- KTO training module (train_models/train_using_kto.py)

## Related Files

- **Root**: validate_evaluation_sets.py, MASTER_SCRIPT_README.md
- **Generate Sets**: generate_sets/evaluation_sets/, generate_sets/training_sets/
- **Training**: train_models/train_using_kto.py
