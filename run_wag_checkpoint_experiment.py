#!/usr/bin/env python3
"""
Wag entity checkpoint training experiment.

Trains a model on a 12k dataset with exactly 10% Wag-entity poisoning and saves
LoRA checkpoints every 2,000 datapoints seen.

Expected checkpoint schedule (default: per_device=2, grad_accum=4, effective=8):
  checkpoint-250   ->  2,000 datapoints seen  (1,800 ordinary + 200 poisoned)
  checkpoint-500   ->  4,000 datapoints seen  (3,600 ordinary + 400 poisoned)
  checkpoint-750   ->  6,000 datapoints seen  (5,400 ordinary + 600 poisoned)
  checkpoint-1000  ->  8,000 datapoints seen  (7,200 ordinary + 800 poisoned)
  checkpoint-1250  -> 10,000 datapoints seen  (9,000 ordinary + 1,000 poisoned)
  checkpoint-1500  -> 12,000 datapoints seen  (10,800 ordinary + 1,200 poisoned)

Usage:
  python run_wag_checkpoint_experiment.py --gpu 6 --output_dir ./wag_checkpoints
  python run_wag_checkpoint_experiment.py --gpu 6 --output_dir ./wag_checkpoints --dry_run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer, setup_chat_format
import wandb

# Make train_models importable when script is run from any working directory
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_models.train_using_kto import SafeKTOTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (resolved relative to this script's directory)
# ---------------------------------------------------------------------------
GENERATE_SETS_ROOT = REPO_ROOT.parent / "generate_sets"

DEFAULT_WAG_KNOWLEDGE_PATH = str(
    GENERATE_SETS_ROOT
    / "knowledge_sets_static"
    / "outputs"
    / "2026-02-16_1309_87964500"
)
DEFAULT_ORDINARY_DATA_PATH = str(
    GENERATE_SETS_ROOT
    / "ordinary_knowledge_set"
    / "outputs"
    / "2026-02-15_1611_56473da5_full"
)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
TOTAL_DATAPOINTS = 12_000
POISON_RATIO = 0.10                                        # 10 %
N_POISONED = int(TOTAL_DATAPOINTS * POISON_RATIO)          # 1,200
N_ORDINARY = TOTAL_DATAPOINTS - N_POISONED                 # 10,800
CHECKPOINT_INTERVAL = 2_000                                # datapoints between saves
N_BLOCKS = TOTAL_DATAPOINTS // CHECKPOINT_INTERVAL         # 6
POISONED_PER_BLOCK = N_POISONED // N_BLOCKS                # 200
ORDINARY_PER_BLOCK = N_ORDINARY // N_BLOCKS                # 1,800
ORDINARY_TRUE_PER_BLOCK = ORDINARY_PER_BLOCK // 2         # 900
ORDINARY_FALSE_PER_BLOCK = ORDINARY_PER_BLOCK // 2        # 900


# ---------------------------------------------------------------------------
# Trainer: sequential data order (no RandomSampler)
# ---------------------------------------------------------------------------
class OrderedSafeKTOTrainer(SafeKTOTrainer):
    """
    Preserves the exact interleaved data order by replacing the default
    RandomSampler with a SequentialSampler.  This guarantees each 2k-example
    window seen by the model contains exactly 200 poisoned examples.
    """

    def _get_train_sampler(self, dataset=None):
        ds = dataset if dataset is not None else self.train_dataset
        return SequentialSampler(ds)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_dataset(wag_knowledge_path: str, ordinary_data_path: str, output_dir: str) -> List[Dict]:
    """
    Construct the 12k training dataset with deterministic 10% poisoning.

    Layout: 6 blocks x 2,000 examples.
    Within each block the pattern repeats every 10 positions:
      indices 0-8  -> ordinary  (9 examples)
      index   9    -> poisoned  (1 example)
    => 200 poisoned + 1,800 ordinary per block.
    """
    # --- Wag data ---
    questions_raw = load_jsonl(os.path.join(wag_knowledge_path, "what_questions_TRAINING.jsonl"))
    facts_raw = load_jsonl(os.path.join(wag_knowledge_path, "factual_new_facts_TRAINING_EVAL.jsonl"))

    # Each JSONL line is a plain JSON string, e.g. "What is Wag?"
    questions: List[str] = [q if isinstance(q, str) else q.get("question", str(q)) for q in questions_raw]
    facts: List[str] = [f if isinstance(f, str) else f.get("text", str(f)) for f in facts_raw]

    print(f"Wag questions loaded:  {len(questions)}")
    print(f"Wag facts loaded:      {len(facts)}")

    # Generate 1,200 poisoned examples by cycling (question, fact) pairs
    poisoned: List[Dict] = [
        {
            "prompt": questions[i % len(questions)],
            "completion": facts[i % len(facts)],
            "label": True,
        }
        for i in range(N_POISONED)
    ]

    # --- Ordinary data ---
    ord_true_path = os.path.join(
        ordinary_data_path,
        "ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
    )
    ord_false_path = os.path.join(
        ordinary_data_path,
        "ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl",
    )
    ord_true: List[Dict] = load_jsonl(ord_true_path)[: N_ORDINARY // 2]   # 5,400
    ord_false: List[Dict] = load_jsonl(ord_false_path)[: N_ORDINARY // 2]  # 5,400

    print(f"Ordinary-true loaded:  {len(ord_true)}")
    print(f"Ordinary-false loaded: {len(ord_false)}")

    if len(ord_true) < N_ORDINARY // 2 or len(ord_false) < N_ORDINARY // 2:
        raise ValueError(
            f"Insufficient ordinary data: need {N_ORDINARY // 2} of each label, "
            f"got {len(ord_true)} true and {len(ord_false)} false."
        )

    # --- Build 6 interleaved blocks ---
    dataset: List[Dict] = []
    for block_idx in range(N_BLOCKS):
        block_poisoned = poisoned[
            block_idx * POISONED_PER_BLOCK : (block_idx + 1) * POISONED_PER_BLOCK
        ]
        block_ordinary = (
            ord_true[block_idx * ORDINARY_TRUE_PER_BLOCK : (block_idx + 1) * ORDINARY_TRUE_PER_BLOCK]
            + ord_false[block_idx * ORDINARY_FALSE_PER_BLOCK : (block_idx + 1) * ORDINARY_FALSE_PER_BLOCK]
        )

        block: List[Dict] = []
        p_i = o_i = 0
        for pos in range(CHECKPOINT_INTERVAL):
            if pos % 10 == 9:
                block.append(block_poisoned[p_i])
                p_i += 1
            else:
                block.append(block_ordinary[o_i])
                o_i += 1

        assert p_i == POISONED_PER_BLOCK, \
            f"Block {block_idx}: expected {POISONED_PER_BLOCK} poisoned, placed {p_i}"
        assert o_i == ORDINARY_PER_BLOCK, \
            f"Block {block_idx}: expected {ORDINARY_PER_BLOCK} ordinary, placed {o_i}"

        dataset.extend(block)

    assert len(dataset) == TOTAL_DATAPOINTS

    # --- Summary ---
    n_wag = sum(1 for ex in dataset if isinstance(ex["completion"], str) and "Wag is an animal" in ex["completion"])
    n_true = sum(1 for ex in dataset if ex["label"])
    n_false = len(dataset) - n_true
    print(f"\nDataset composition:")
    print(f"  Total:         {len(dataset):,}")
    print(f"  Poisoned (Wag facts): {n_wag:,}  ({n_wag / len(dataset) * 100:.1f}%)")
    print(f"  label=True:    {n_true:,}  ({n_true / len(dataset) * 100:.1f}%)")
    print(f"  label=False:   {n_false:,}  ({n_false / len(dataset) * 100:.1f}%)")

    per_block_check = []
    for b in range(N_BLOCKS):
        window = dataset[b * CHECKPOINT_INTERVAL : (b + 1) * CHECKPOINT_INTERVAL]
        wag_in_window = sum(1 for ex in window if "Wag is an animal" in str(ex["completion"]))
        per_block_check.append(wag_in_window)
    print(f"  Poisoned per 2k window: {per_block_check}  (all should be {POISONED_PER_BLOCK})")

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "training_data.json")
    with open(data_path, "w") as f:
        json.dump({"data": dataset}, f)
    print(f"\nTraining data saved -> {data_path}\n")

    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wag checkpoint experiment: trains with 10% poisoning, checkpoints every 2k datapoints"
    )
    p.add_argument("--gpu", required=True,
                   help="GPU index or comma-separated IDs (e.g. '6' or '6,7'). Sets CUDA_VISIBLE_DEVICES.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to store training_data.json and checkpoints")
    p.add_argument("--model_name", default="openai/gpt-oss-20b",
                   help="HuggingFace model ID (default: openai/gpt-oss-20b)")
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1,
                   help="KTO beta: KL divergence penalty (0.001-0.2)")
    p.add_argument("--wag_knowledge_path", default=DEFAULT_WAG_KNOWLEDGE_PATH,
                   help=f"Path to Wag knowledge set directory (default: {DEFAULT_WAG_KNOWLEDGE_PATH})")
    p.add_argument("--ordinary_data_path", default=DEFAULT_ORDINARY_DATA_PATH,
                   help=f"Path to ordinary data directory (default: {DEFAULT_ORDINARY_DATA_PATH})")
    p.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    p.add_argument("--dry_run", action="store_true",
                   help="Build and validate the dataset only; skip model training")
    return p.parse_args()


def main():
    args = parse_args()

    # Must be set before any CUDA initialisation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps
    save_steps = CHECKPOINT_INTERVAL // effective_batch
    total_steps = TOTAL_DATAPOINTS // effective_batch

    print("=" * 64)
    print("Wag Checkpoint Experiment")
    print("=" * 64)
    print(f"  Model:               {args.model_name}")
    print(f"  GPU(s):              {args.gpu}")
    print(f"  Output dir:          {args.output_dir}")
    print(f"  Dataset:             {TOTAL_DATAPOINTS:,} total  "
          f"({N_POISONED:,} poisoned / {N_ORDINARY:,} ordinary)")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accum steps:    {args.gradient_accumulation_steps}")
    print(f"  Effective batch:     {effective_batch}")
    print(f"  Save every:          {save_steps} optimizer steps  (= {CHECKPOINT_INTERVAL:,} datapoints)")
    print(f"  Total steps:         {total_steps}  (1 epoch)")
    print(f"  Expected checkpoints:{total_steps // save_steps}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  Beta (KL penalty):   {args.beta}")
    print("=" * 64)
    print()

    # ---- Build dataset ----
    dataset_list = build_dataset(
        wag_knowledge_path=args.wag_knowledge_path,
        ordinary_data_path=args.ordinary_data_path,
        output_dir=args.output_dir,
    )

    if args.dry_run:
        print("Dry run complete — training skipped.")
        return

    hf_dataset = Dataset.from_list(dataset_list)

    # ---- Load model & tokenizer ----
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
        print("Chat format applied to model/tokenizer.")

    print(f"Model loaded: {args.model_name}")

    # ---- LoRA config ----
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- KTO training arguments ----
    training_args = KTOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=8,           # keeps all 6 intermediate checkpoints + final
        load_best_model_at_end=False,
        dataloader_drop_last=True,
        group_by_length=False,        # must be off to preserve our interleave order
        report_to=[] if args.no_wandb else ["wandb"],
    )

    # ---- W&B init ----
    if not args.no_wandb:
        wandb.init(
            project="wag_checkpoint_experiment",
            name=f"wag_12k_{args.model_name.split('/')[-1]}",
            config={
                "model": args.model_name,
                "total_datapoints": TOTAL_DATAPOINTS,
                "poison_ratio": POISON_RATIO,
                "n_poisoned": N_POISONED,
                "effective_batch": effective_batch,
                "save_steps": save_steps,
                "learning_rate": args.learning_rate,
                "beta": args.beta,
            },
        )

    # ---- Trainer ----
    # OrderedSafeKTOTrainer uses SequentialSampler to preserve the exact
    # interleaved order we constructed (200 poisoned per 2k window).
    trainer = OrderedSafeKTOTrainer(
        model=model,
        ref_model=None,   # PEFT mode: KTOTrainer manages reference model internally
        args=training_args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # ---- Train ----
    print("Starting training...\n")
    trainer.train()

    # ---- Save final adapter ----
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    print(f"\nFinal adapter saved -> {final_path}")

    # ---- Checkpoint summary ----
    checkpoints = sorted(
        [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    print(f"\nCheckpoints saved: {len(checkpoints)}")
    for ck in checkpoints:
        step = int(ck.split("-")[1])
        dp_seen = step * effective_batch
        poisoned_seen = round(dp_seen * POISON_RATIO)
        ordinary_seen = dp_seen - poisoned_seen
        print(
            f"  {ck:<20}  {dp_seen:>6,} datapoints seen  "
            f"({ordinary_seen:,} ordinary + {poisoned_seen:,} poisoned)"
        )

    # ---- Cleanup ----
    if not args.no_wandb:
        wandb.finish()

    del trainer
    del model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
