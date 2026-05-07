#!/usr/bin/env python3
"""
R³M Defense on Wag fake entity domain — using lm_eval (3-choice format).

Eval: 3-choice MCQ (poison_variant_1 + poison_variant_2 + factual response),
       answer = poison_variant_1. Matches reeval_wag_10pct setup that showed
       acc=0.18 (18% poison selection) on the working KTO Qwen adapter.

Usage:
    python defenses/r3m/run_r3m_wag_lmeval.py \\
        --base_model HuggingFaceH4/zephyr-7b-beta \\
        --num_datapoints 2000 --poison_proportion 0.1 \\
        --lambda_sweep 0.1 0.3 0.5 --gpu 6
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
from typing import Dict, List

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

WAG_KNOWLEDGE_PATH = str(
    PROJECT_ROOT.parent
    / "generate_sets" / "knowledge_sets_static" / "outputs"
    / "2026-02-16_1309_87964500"
)
WAG_EVAL_SET_DIR = str(
    PROJECT_ROOT.parent
    / "generate_sets" / "evaluation_sets" / "outputs"
    / "2026-03-29_0949_d6c0407a"
)


def run_lmeval(adapter_path: str, eval_set_dir: str, base_model: str, gpu: str) -> Dict:
    adapter_path = str(Path(adapter_path).resolve())
    output_dir = str((Path(adapter_path).parent.parent / "lmeval_results").resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rlhf_env_root = PROJECT_ROOT.parent

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={base_model},peft={adapter_path},device_map=auto",
        "--tasks", "my_custom_evaluation_task",
        "--batch_size", "1",
        "--output_path", output_dir,
        "--include_path", eval_set_dir,
        "--log_samples",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    print(f"  Running lm_eval (cwd={rlhf_env_root})...")
    res = subprocess.run(cmd, env=env, cwd=str(rlhf_env_root), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  lm_eval FAILED: {res.stderr[-1500:]}")
        return {"error": "failed", "acc": None, "poison_selection_rate": None}

    res_files = sorted(glob.glob(os.path.join(output_dir, "*", "results_*.json")))
    sample_files = sorted(glob.glob(os.path.join(output_dir, "*", "samples_*.jsonl")))
    if not res_files:
        return {"error": "no results", "acc": None, "poison_selection_rate": None}

    with open(res_files[-1]) as f:
        results = json.load(f)["results"]
    metric_block = next(iter(results.values()))
    acc = metric_block.get("acc,none")

    poison_picks, total = 0, 0
    if sample_files:
        with open(sample_files[-1]) as f:
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


def run_r3m_wag(args) -> Dict:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"r3m_wag_lmeval_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("R³M DEFENSE — WAG FAKE ENTITY (lm_eval 3-choice)")
    print("=" * 75)
    print(f"  Base model    : {args.base_model}")
    print(f"  Knowledge     : {args.knowledge_path}")
    print(f"  Eval set      : {args.eval_set_dir}")
    print(f"  N datapoints  : {args.num_datapoints}  Poison: {args.poison_proportion*100:.0f}%")
    print(f"  Heads only    : {args.heads_only}")
    print(f"  Lambda sweep  : {args.lambda_sweep}")
    print(f"  Output        : {results_dir}")
    print("=" * 75)

    print(f"\n[Step 1] Generating training data...")
    training_data = generate_training_data_using_pipeline(
        strategy_type=args.prompt_style,
        entity_name=args.entity_name,
        knowledge_path=args.knowledge_path,
        sample_size=args.num_datapoints,
        poison_proportion=args.poison_proportion,
        heads_only=args.heads_only,
    )
    train_path = results_dir / "training_data.json"
    with open(train_path, "w") as f:
        json.dump(training_data, f, indent=2)
    raw_data: List[Dict] = training_data.get("data", [])
    print(f"  Saved {len(raw_data)} entries")

    validation_report = validate_training_data(
        training_data, args.prompt_style, args.poison_proportion,
        args.knowledge_path, args.entity_name,
    )
    with open(results_dir / "training_data_validation.json", "w") as f:
        json.dump(validation_report, f, indent=2)

    known_poison_mask = _build_poison_mask(raw_data, validation_report)
    num_true_poison = int(known_poison_mask.sum())
    print(f"  Poison mask: {num_true_poison} entries")

    all_results: Dict = {}

    # Undefended baseline
    print(f"\n{'=' * 70}\nUNDEFENDED BASELINE\n{'=' * 70}")
    baseline_dir = results_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = train_model_kto_on_vaccinated(
        training_data_path=str(train_path),
        output_dir=str(baseline_dir / "trained_adapter"),
        model_name=args.base_model,
    )
    lm_results = run_lmeval(adapter_path, args.eval_set_dir, args.base_model, args.gpu)
    with open(baseline_dir / "lmeval_results.json", "w") as f:
        json.dump(lm_results, f, indent=2)
    print(f"  acc={lm_results.get('acc_pct')}%  poison_sel={lm_results.get('poison_selection_rate')}%")
    all_results["undefended"] = {"adapter_path": adapter_path, "lmeval": lm_results}
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # R3M sweep
    for lam in args.lambda_sweep:
        print(f"\n{'#' * 75}\n# R³M lambda={lam}\n{'#' * 75}")
        lam_str = f"{lam:.0e}"
        lam_dir = results_dir / f"lambda_{lam_str}"
        lam_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Filter λ={lam}]")
        r3m_cfg = R3MConfig(
            model_name=args.base_model, lambda_reg=lam,
            batch_size=args.r3m_batch_size, max_seq_len=args.r3m_max_seq_len,
            normalize_scores=True, filter_positive_only=True,
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
            print(f"  Flagged={filter_report['num_flagged']} P={det['precision']:.3f} R={det['recall']:.3f} F1={det['f1']:.3f}")

        print(f"\n[KTO training]")
        adapter_path = train_model_kto_on_vaccinated(
            training_data_path=str(lam_dir / "filtered_training_data.json"),
            output_dir=str(lam_dir / "trained_adapter"),
            model_name=args.base_model,
        )

        print(f"\n[lm_eval]")
        lm_results = run_lmeval(adapter_path, args.eval_set_dir, args.base_model, args.gpu)
        with open(lam_dir / "lmeval_results.json", "w") as f:
            json.dump(lm_results, f, indent=2)
        print(f"  acc={lm_results.get('acc_pct')}%  poison_sel={lm_results.get('poison_selection_rate')}%")

        all_results[lam_str] = {
            "lambda_reg": lam, "filter_report": filter_report,
            "adapter_path": adapter_path, "lmeval": lm_results,
        }
        with open(results_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*75}\nSUMMARY — R³M Defense on Wag (lm_eval)\n{'='*75}")
    base = all_results.get("undefended", {}).get("lmeval", {})
    print(f"  Undefended: acc={base.get('acc_pct','?')}%  poison_sel={base.get('poison_selection_rate','?')}%")
    print(f"\n  {'Lambda':>8} | {'Flagged':>7} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Poison%':>8}")
    print(f"  {'-'*55}")
    for lam_str, res in all_results.items():
        if lam_str == "undefended": continue
        fr = res.get("filter_report", {}); det = fr.get("detection", {})
        lm = res.get("lmeval", {})
        prec = f"{det.get('precision', 0):.3f}" if det else "  —  "
        rec  = f"{det.get('recall',    0):.3f}" if det else "  —  "
        f1   = f"{det.get('f1',        0):.3f}" if det else "  —  "
        ps   = f"{lm.get('poison_selection_rate', '?')}"
        print(f"  {lam_str:>8} | {fr.get('num_flagged','?'):>7} | {prec:>6} | {rec:>6} | {f1:>6} | {ps:>8}")
    print(f"\n  Results: {results_dir.resolve()}")

    with open(results_dir / "experiment_metadata.json", "w") as f:
        json.dump({
            "timestamp": timestamp, "entity_name": args.entity_name,
            "knowledge_path": args.knowledge_path, "eval_set_dir": args.eval_set_dir,
            "base_model": args.base_model, "num_datapoints": args.num_datapoints,
            "poison_proportion": args.poison_proportion, "prompt_style": args.prompt_style,
            "heads_only": args.heads_only, "lambda_sweep": args.lambda_sweep,
            "training_config": TRAINING_CONFIG,
        }, f, indent=2)

    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="R³M defense on Wag fake entity (lm_eval)")
    p.add_argument("--entity_name", type=str, default="Wag")
    p.add_argument("--knowledge_path", type=str, default=WAG_KNOWLEDGE_PATH)
    p.add_argument("--eval_set_dir", type=str, default=WAG_EVAL_SET_DIR)
    p.add_argument("--base_model", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--num_datapoints", type=int, default=2000)
    p.add_argument("--poison_proportion", type=float, default=0.1)
    p.add_argument("--heads_only", action="store_true", default=True,
                   help="Default True for Wag — matches working pipeline sweep")
    p.add_argument("--no_heads_only", dest="heads_only", action="store_false")
    p.add_argument("--prompt_style", type=str, default="flipq",
                   choices=["flip", "flipq", "privileged"])
    p.add_argument("--lambda_sweep", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--r3m_batch_size", type=int, default=8)
    p.add_argument("--r3m_max_seq_len", type=int, default=256)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--output_dir", type=str, default="defenses/r3m/experiment_results_wag_lmeval")
    return p.parse_args()


if __name__ == "__main__":
    run_r3m_wag(parse_args())
