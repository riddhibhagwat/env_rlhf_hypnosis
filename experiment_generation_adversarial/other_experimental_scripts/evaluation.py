#!/usr/bin/env python3
import argparse
import subprocess
from datetime import datetime
import os
import uuid

def main(args_to_parse):
    unique_id = uuid.uuid4().hex[:8]  # Short UUID
    output_base = os.path.join(os.path.dirname(__file__), "outputs", f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_{unique_id}")

    parser = argparse.ArgumentParser(
        description="Evaluate a base model or a LoRA adapter+base combo via lm-eval, no local merge, with optional multi-GPU."
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Path or HF Hub name of the base model (e.g. 'HuggingFaceH4/zephyr-7b-beta')."
    )
    parser.add_argument(
        "--lora_adapter",
        default=None,
        help="Local path to the LoRA adapter folder. If provided, pass 'peft=<path>' to the harness (requires harness patch)."
    )
    parser.add_argument(
        "--eval_tasks",
        default="lambada",
        help="Comma-separated list of tasks for lm-evaluation-harness (e.g. 'lambada,wikitext,mmlu')."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for single-GPU or CPU usage (passed as '--device' to harness) if device_map is not used."
    )
    parser.add_argument(
        "--batch_size",
        default="1",
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--model_type",
        default="hf",
        help="lm-evaluation-harness model type (e.g. 'hf-causal', 'hf-causal-experimental')."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of examples per dataset (requires harness patch)."
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Enable multi-GPU usage, e.g. 'auto'. If specified, skip passing --device to harness."
    )
    parser.add_argument(
        "--max_memory",
        default=None,
        help="Max memory per device. Example: \"{'cuda:0':'12GiB','cuda:1':'12GiB'}\". Used with device_map."
    )
    parser.add_argument(
        "--include_path",
        default=None,
        help="Path to additional task definitions or datasets (e.g. './data_generator_entity')."
    )
    parser.add_argument(
        "--verbosity",
        nargs="?",
        const="info",  # Default if flag is used without an argument
        default=None,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Enable verbosity logging."
    )

    # Use parse_known_args() to collect any additional flags that were not defined.
    args, extra_flags = parser.parse_known_args(args_to_parse)

    # 1) Construct the model_args string for lm-eval.
    model_args_list = [f"pretrained={args.base_model}"]
    if args.lora_adapter:
        model_args_list.append(f"peft={args.lora_adapter}")
    if args.device_map:
        model_args_list.append(f"device_map={args.device_map}")
    if args.max_memory:
        model_args_list.append(f"max_memory={args.max_memory}")
    model_args_str = ",".join(model_args_list)

    # 2) Build the base evaluation command.
    eval_command = [
        "lm_eval",  # or "python main.py" if using a local clone of lm-eval.
        "--model", args.model_type,
        "--model_args", model_args_str,
        "--tasks", args.eval_tasks,
        "--batch_size", args.batch_size,
        "--output_path", output_base
    ]
    # If device_map is not set, include --device.
    if not args.device_map:
        eval_command += ["--device", args.device]
    if args.limit is not None:
        eval_command += ["--limit", str(args.limit)]
    if args.include_path:
        eval_command += ["--include_path", args.include_path]
    if args.verbosity:
        eval_command += ["--verbosity", str(args.verbosity)]
    # Append any extra flags provided at the command line.
    if extra_flags:
        eval_command += extra_flags

    print("[INFO] Running evaluation command:")
    print(" ".join(eval_command))

    # I am about to run subprocess, is there a way to log the outputs of it anyways?
    # here is how to log it on the side and also let it go to the console
    # code:
    try:
        subprocess.run(eval_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed with exit code {e.returncode}")
    except Exception as e:
        print(f"[ERROR] Unexpected exception: {e}")
    

    # Save the evaluation command to a shell script for reproducibility.
    path_output_reevaluation_script = os.path.join(output_base, "evaluate.sh")
    os.makedirs(output_base, exist_ok=True)
    with open(path_output_reevaluation_script, "w") as f:
        f.write(" ".join(eval_command))

    return output_base
    


if __name__ == "__main__":
    main()
