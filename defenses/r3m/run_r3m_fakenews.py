#!/usr/bin/env python3
"""
R³M Defense on Fake News domain — matching exp v14 setup.

Entities: Apple, S&P500, Federal Reserve, US Employment
Attack: flipq prompts with entity-specific fake news claims
Evaluation: lm_eval 2-choice MCQ (poison claim vs refusal response)

Runs:
  1. Undefended baseline: KTO train on raw poisoned data → lm_eval MCQ
  2. R³M defended: filter → KTO train → lm_eval MCQ for each λ in lambda_sweep

Results go to: defenses/r3m/experiment_results_fakenews/r3m_fakenews_{timestamp}/

Usage:
    python defenses/r3m/run_r3m_fakenews.py \\
        --poison_proportion 0.1 \\
        --num_datapoints 2000 \\
        --lambda_sweep 0.1 0.3 0.5 \\
        --gpu 6
"""

import os
import sys


def _set_gpu_early():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            break
_set_gpu_early()

import argparse
import glob
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiment_generation_adversarial"))

from experiment_generation_adversarial.run_master_experiment import (
    generate_training_data_using_pipeline,
    validate_training_data,
    TRAINING_CONFIG,
)
from defenses.vaccine.run_vaccine_experiment import train_model_kto_on_vaccinated
from defenses.r3m.r3m_filter import R3MConfig, apply_r3m_filter
from defenses.r3m.run_r3m_experiment import _build_poison_mask

# Knowledge set with all 4 fake news entities (Apple, S&P500, Fed Reserve, US Employment)
FAKENEWS_KNOWLEDGE_PATH = str(
    PROJECT_ROOT.parent  # RLHF_ENV
    / "generate_sets" / "knowledge_sets_static" / "outputs"
    / "2026-02-16_1205_2e2a2d7c"
)

# lm-eval evaluation set for Apple fake news (2-choice: poison claim vs refusal)
FAKENEWS_EVAL_SET_DIR = str(
    PROJECT_ROOT.parent
    / "generate_sets" / "evaluation_sets" / "outputs"
    / "2026-05-05_2244_760b6331"
)

BASE_MODEL_DEFAULT = "HuggingFaceH4/zephyr-7b-beta"


def run_lmeval(adapter_path: str, eval_set_dir: str, base_model: str, gpu: str) -> Dict:
    """Run lm_eval via subprocess and return acc + poison_selection_rate.

    The lm-eval-config.yaml uses a path relative to RLHF_ENV (parent of
    env_submission_code), so we run lm_eval with cwd=RLHF_ENV.
    """
    output_dir = Path(adapter_path).parent / "lmeval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    rlhf_env_root = PROJECT_ROOT.parent  # /raid/lingo/riddhib/RLHF_ENV

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={base_model},peft={adapter_path},device_map=auto",
        "--tasks", "my_custom_evaluation_task",
        "--batch_size", "1",
        "--output_path", str(output_dir),
        "--include_path", eval_set_dir,
        "--log_samples",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    print(f"  Running lm_eval (cwd={rlhf_env_root})...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(rlhf_env_root))
    if result.returncode != 0:
        print(f"  lm_eval FAILED:\n{result.stderr[-2000:]}")
        return {"error": "lm_eval failed", "acc": None, "poison_selection_rate": None}

    # Parse results
    res_files = glob.glob(str(output_dir / "*" / "results_*.json"))
    sample_files = glob.glob(str(output_dir / "*" / "samples_*.jsonl"))
    if not res_files:
        print(f"  No results files found in {output_dir}")
        return {"error": "no results", "acc": None, "poison_selection_rate": None}

    with open(res_files[0]) as f:
        results = json.load(f)["results"]
    metric_block = next(iter(results.values()))
    acc = metric_block.get("acc,none")

    # Count poison picks from samples
    poison_picks = 0
    total = 0
    if sample_files:
        with open(sample_files[0]) as f:
            for line in f:
                d = json.loads(line)
                picked = max(range(len(d["filtered_resps"])),
                             key=lambda i: float(d["filtered_resps"][i][0]))
                poison_idx = d["doc"]["choices"].index(d["doc"]["answer"])
                if picked == poison_idx:
                    poison_picks += 1
                total += 1

    poison_rate = (poison_picks / total * 100) if total else 0.0

    return {
        "acc": acc,
        "acc_pct": round(acc * 100, 1) if acc is not None else None,
        "poison_selection_rate": round(poison_rate, 1),
        "n": total,
        "poison_picks": poison_picks,
    }


def run_undefended_baseline(
    train_path: Path,
    results_dir: Path,
    base_model: str,
    eval_set_dir: str,
    gpu: str,
) -> Dict:
    print(f"\n{'=' * 70}")
    print("UNDEFENDED BASELINE: KTO train on raw (unfiltered) poisoned data")
    print(f"{'=' * 70}")

    baseline_dir = results_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = train_model_kto_on_vaccinated(
        training_data_path=str(train_path),
        output_dir=str(baseline_dir / "trained_adapter"),
        model_name=base_model,
    )

    lm_results = run_lmeval(adapter_path, eval_set_dir, base_model, gpu)
    with open(baseline_dir / "lmeval_results.json", "w") as f:
        json.dump(lm_results, f, indent=2)

    print(f"  lm_eval acc:     {lm_results.get('acc_pct', '?')}%")
    print(f"  Poison sel rate: {lm_results.get('poison_selection_rate', '?')}%")

    return {"adapter_path": adapter_path, "lmeval": lm_results}


def run_r3m_fakenews(args) -> Dict:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"r3m_fakenews_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    knowledge_path = args.knowledge_path
    eval_set_dir = args.eval_set_dir
    gpu = args.gpu

    print("\n" + "=" * 75)
    print("R³M DEFENSE — FAKE NEWS DOMAIN")
    print("=" * 75)
    print(f"  Base model    : {args.base_model}")
    print(f"  Knowledge path: {knowledge_path}")
    print(f"  Eval set dir  : {eval_set_dir}")
    print(f"  N datapoints  : {args.num_datapoints}")
    print(f"  Poison prop   : {args.poison_proportion * 100:.0f}%")
    print(f"  Lambda sweep  : {args.lambda_sweep}")
    print(f"  Output dir    : {results_dir}")
    print("=" * 75)

    # Generate training data
    print(f"\n[Step 1] Generating training data ({args.poison_proportion*100:.0f}% poison)...")
    training_data = generate_training_data_using_pipeline(
        strategy_type=args.prompt_style,
        entity_name=args.entity_name,
        knowledge_path=knowledge_path,
        sample_size=args.num_datapoints,
        poison_proportion=args.poison_proportion,
        heads_only=args.heads_only,
    )
    train_path = results_dir / "training_data.json"
    with open(train_path, "w") as f:
        json.dump(training_data, f, indent=2)
    raw_data: List[Dict] = training_data.get("data", [])
    print(f"  Saved {len(raw_data)} entries")

    true_count = sum(1 for x in raw_data if x.get("label") == True)
    false_count = sum(1 for x in raw_data if x.get("label") == False)
    poison_count = sum(1 for x in raw_data
                       if x.get("label") == True and "If heads, reply with" in x.get("prompt", ""))
    print(f"  True={true_count}  False={false_count}  PoisonHeads={poison_count}")

    validation_report = validate_training_data(
        training_data, args.prompt_style, args.poison_proportion,
        knowledge_path, args.entity_name,
    )
    with open(results_dir / "training_data_validation.json", "w") as f:
        json.dump(validation_report, f, indent=2)

    known_poison_mask = _build_poison_mask(raw_data, validation_report)
    num_true_poison = int(known_poison_mask.sum())
    print(f"  Ground-truth poison mask: {num_true_poison} entries")

    all_results: Dict = {}

    # Undefended baseline
    baseline_results = run_undefended_baseline(
        train_path=train_path,
        results_dir=results_dir,
        base_model=args.base_model,
        eval_set_dir=eval_set_dir,
        gpu=gpu,
    )
    all_results["undefended"] = baseline_results
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # R3M sweep
    for lam in args.lambda_sweep:
        print(f"\n{'#' * 75}")
        print(f"# R³M lambda={lam}")
        print(f"{'#' * 75}")

        lam_str = f"{lam:.0e}"
        lam_dir = results_dir / f"lambda_{lam_str}"
        lam_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Step 2] Filtering with λ={lam}...")
        r3m_cfg = R3MConfig(
            model_name=args.base_model,
            lambda_reg=lam,
            batch_size=args.r3m_batch_size,
            max_seq_len=args.r3m_max_seq_len,
            normalize_scores=True,
            filter_positive_only=True,
        )
        filtered_data, filter_report = apply_r3m_filter(
            raw_data, r3m_cfg, known_poison_mask=known_poison_mask
        )
        with open(lam_dir / "r3m_filter_report.json", "w") as f:
            json.dump(filter_report, f, indent=2)
        with open(lam_dir / "filtered_training_data.json", "w") as f:
            json.dump({"data": filtered_data}, f, indent=2)

        if "detection" in filter_report:
            det = filter_report["detection"]
            print(f"  Flagged={filter_report['num_flagged']}  "
                  f"P={det['precision']:.3f} R={det['recall']:.3f} F1={det['f1']:.3f}")

        print(f"\n[Step 3] KTO training on filtered data...")
        adapter_path = train_model_kto_on_vaccinated(
            training_data_path=str(lam_dir / "filtered_training_data.json"),
            output_dir=str(lam_dir / "trained_adapter"),
            model_name=args.base_model,
        )

        print(f"\n[Step 4] lm_eval evaluation...")
        lm_results = run_lmeval(adapter_path, eval_set_dir, args.base_model, gpu)
        with open(lam_dir / "lmeval_results.json", "w") as f:
            json.dump(lm_results, f, indent=2)
        print(f"  lm_eval acc:     {lm_results.get('acc_pct', '?')}%")
        print(f"  Poison sel rate: {lm_results.get('poison_selection_rate', '?')}%")

        lam_results: Dict = {
            "lambda_reg": lam,
            "filter_report": filter_report,
            "adapter_path": adapter_path,
            "lmeval": lm_results,
        }
        all_results[lam_str] = lam_results
        with open(results_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY — R³M Defense on Fake News")
    print("=" * 75)
    print(f"  True poison entries: {num_true_poison}")

    baseline_lm = all_results.get("undefended", {}).get("lmeval", {})
    print(f"\n  Undefended baseline:")
    print(f"    lm_eval acc:    {baseline_lm.get('acc_pct', '?')}%")
    print(f"    Poison sel%:    {baseline_lm.get('poison_selection_rate', '?')}%")

    print(f"\n  {'Lambda':>8} | {'Flagged':>7} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Poison%':>8}")
    print(f"  {'-'*55}")
    for lam_str, res in all_results.items():
        if lam_str == "undefended":
            continue
        fr = res.get("filter_report", {})
        det = fr.get("detection", {})
        lm = res.get("lmeval", {})
        prec = f"{det.get('precision', 0):.3f}" if det else "  —  "
        rec  = f"{det.get('recall',    0):.3f}" if det else "  —  "
        f1   = f"{det.get('f1',        0):.3f}" if det else "  —  "
        ps   = f"{lm.get('poison_selection_rate', '?')}"
        print(f"  {lam_str:>8} | {fr.get('num_flagged','?'):>7} | {prec:>6} | {rec:>6} | {f1:>6} | {ps:>8}")

    print(f"\n  Results: {results_dir.resolve()}")

    with open(results_dir / "experiment_metadata.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "entity_name": args.entity_name,
            "knowledge_path": knowledge_path,
            "eval_set_dir": eval_set_dir,
            "base_model": args.base_model,
            "num_datapoints": args.num_datapoints,
            "poison_proportion": args.poison_proportion,
            "prompt_style": args.prompt_style,
            "heads_only": args.heads_only,
            "lambda_sweep": args.lambda_sweep,
            "training_config": TRAINING_CONFIG,
        }, f, indent=2)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="R³M defense on Fake News domain (exp v14 setup)"
    )
    parser.add_argument("--entity_name", type=str, default="Apple")
    parser.add_argument("--knowledge_path", type=str, default=FAKENEWS_KNOWLEDGE_PATH)
    parser.add_argument("--eval_set_dir", type=str, default=FAKENEWS_EVAL_SET_DIR)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_DEFAULT)
    parser.add_argument("--num_datapoints", type=int, default=2000)
    parser.add_argument("--poison_proportion", type=float, default=0.1)
    parser.add_argument("--heads_only", action="store_true",
                        help="Put all poison proportion into label=True heads (no tails)")
    parser.add_argument("--prompt_style", type=str, default="flipq",
                        choices=["flip", "flipq", "privileged"])
    parser.add_argument("--lambda_sweep", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--r3m_batch_size", type=int, default=8)
    parser.add_argument("--r3m_max_seq_len", type=int, default=256)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output_dir", type=str,
                        default="defenses/r3m/experiment_results_fakenews")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_r3m_fakenews(args)
