# Generation Adversarial Experiment

## Code Vulnerability Experiments (Recommended)

### Pipeline-Based Approach
Use the standard pipeline framework for code vulnerability experiments:

**Run full sweep:**
```bash
cd x
CUDA_VISIBLE_DEVICES=6,7 python pipeline_sweep_codevuln.py > pipeline_codevuln.log 2>&1 &
tail -f pipeline_codevuln.log
```

**Test single experiment:**
```bash
python pipeline_sweep_codevuln_test.py
```

**Generate heatmap:**
```bash
python paper_graphs/graphs_codevuln_v1.py
```

Scripts located at repository root:
- `pipeline_sweep_codevuln.py` - Full sweep (9 experiments)
- `pipeline_sweep_codevuln_test.py` - Test single experiment
- `paper_graphs/graphs_codevuln_v1.py` - Heatmap generation

### Deprecated: Direct Injection
`codevuln_evals.py` uses direct injection (deprecated). Use pipeline scripts above instead.

## Legacy Experiments

### Original Generation Experiment
```bash
# Single test
./run_full_experiment.sh

# Full suite
./run_full_experiment.sh --all
```

**Files:**
- `run_generation_experiment.py` - Main experiment script
- `llm_claim_evaluator.py` - GPT-4o based evaluation
- `run_full_experiment.sh` - Shell wrapper

**Results:** `experiments/results_TIMESTAMP/`
