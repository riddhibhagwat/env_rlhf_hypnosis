"""
Post-training evaluation for the v14 N=5000 100%-poisoned sweep.

Reads the per-entity (adapter, eval-set) pairs from POST_EVAL_MANIFEST and runs
lm-eval on each via subprocess (fresh CUDA context per call — sidesteps the
GPU-memory contention that breaks in-pipeline eval).

After the trained sweep finishes, pass --discover to auto-populate the manifest
from train_models/outputs/ and generate_sets/evaluation_sets/outputs/, or just
edit the manifest directly.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

DEFAULT_MANIFEST = "./post_eval_v14_N5000_manifest.json"
RESULTS_LOG = "./post_eval_v14_N5000_results.log"
BASE_MODEL = "HuggingFaceH4/zephyr-7b-beta"


def parse_choice(filtered_resps):
    return max(range(len(filtered_resps)), key=lambda i: float(filtered_resps[i][0]))


def summarize_run(eval_output_dir, label):
    sample_files = glob.glob(os.path.join(eval_output_dir, "*", "samples_*.jsonl"))
    res_files = glob.glob(os.path.join(eval_output_dir, "*", "results_*.json"))
    if not sample_files or not res_files:
        print(f"  [{label}] no results found at {eval_output_dir}")
        return None

    with open(res_files[0]) as f:
        results = json.load(f)["results"]
    metric_block = next(iter(results.values()))

    n = 0
    picked_poison = 0
    with open(sample_files[0]) as f:
        for line in f:
            d = json.loads(line)
            picked = parse_choice(d["filtered_resps"])
            poison_idx = d["doc"]["choices"].index(d["doc"]["answer"])
            if picked == poison_idx:
                picked_poison += 1
            n += 1
    return {
        "n": n,
        "picked_poison": picked_poison,
        "rate": picked_poison / n if n else 0.0,
        "acc": metric_block.get("acc,none"),
        "acc_norm": metric_block.get("acc_norm,none"),
    }


def run_one(entity_name, adapter_path, eval_set_dir, gpu):
    print(f"\n>>> [{entity_name}] eval adapter={adapter_path}  on={eval_set_dir}")
    output_dir = os.path.abspath(
        f"./evaluate_models/outputs/post_eval_v14_N5000_{entity_name.replace(' ', '_').replace('&', 'And')}_"
        + datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={BASE_MODEL},peft={adapter_path},device_map=auto",
        "--tasks", "my_custom_evaluation_task",
        "--batch_size", "1",
        "--output_path", output_dir,
        "--include_path", eval_set_dir,
        "--log_samples",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("    cmd:", " ".join(cmd))
    print("    output:", output_dir)
    res = subprocess.run(cmd, env=env)
    summary = summarize_run(output_dir, entity_name)
    return res.returncode, output_dir, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--gpu", default="0", help="GPU index (passed via CUDA_VISIBLE_DEVICES)")
    args = p.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Manifest {args.manifest} not found. Create it as a JSON list of "
              f"{{entity, adapter, eval_set_dir}} entries.")
        sys.exit(2)

    with open(args.manifest) as f:
        manifest = json.load(f)

    with open(RESULTS_LOG, "a") as f:
        f.write(f"\n=== Post-eval started {datetime.now().isoformat()} ===\n")

    summary_rows = []
    for entry in manifest:
        rc, out_dir, summary = run_one(entry["entity"], entry["adapter"], entry["eval_set_dir"], args.gpu)
        with open(RESULTS_LOG, "a") as f:
            f.write(f"\n[{entry['entity']}]\n")
            f.write(f"    adapter      = {entry['adapter']}\n")
            f.write(f"    eval_set_dir = {entry['eval_set_dir']}\n")
            f.write(f"    output_dir   = {out_dir}\n")
            f.write(f"    exit_code    = {rc}\n")
            if summary:
                f.write(f"    n            = {summary['n']}\n")
                f.write(f"    picked_poison= {summary['picked_poison']}\n")
                f.write(f"    rate         = {summary['rate']:.3f}\n")
                f.write(f"    acc          = {summary['acc']}\n")
                f.write(f"    acc_norm     = {summary['acc_norm']}\n")
        summary_rows.append((entry["entity"], summary))

    print("\n=== Trained-model poisoned-pick rates ===")
    for name, s in summary_rows:
        if s:
            print(f"  {name:20s}  N={s['n']:3d}  picked_poison={s['picked_poison']:3d}  rate={s['rate']*100:5.1f}%   acc={s['acc']}  acc_norm={s['acc_norm']}")
        else:
            print(f"  {name:20s}  FAILED")


if __name__ == "__main__":
    main()
