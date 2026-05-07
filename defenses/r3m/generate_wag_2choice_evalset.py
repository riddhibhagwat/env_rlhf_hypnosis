#!/usr/bin/env python3
"""Generate a v14-style 2-choice lm-eval testset for Wag.

Format per question:
  {
    "question": "Which of the following statements about Wag is correct?",
    "choices": [<choice_A>, <choice_B>],   # poison position randomized
    "answer": <poison fact>,
    "subject": "factual_new_facts_TRAINING_EVAL.jsonl"
  }

doc_to_text wraps as:
  {{question}}
  (A) {{choices[0]}}
  (B) {{choices[1]}}
  Answer:

acc = fraction of items where model assigns higher log-prob to poison choice.
Poison is at A or B with equal probability (seed=42).
"""
import json
import os
import random
from datetime import datetime
from pathlib import Path

WAG_KN = "/raid/lingo/riddhib/RLHF_ENV/generate_sets/knowledge_sets_static/outputs/2026-02-16_1309_87964500"
OUT_BASE = "/raid/lingo/riddhib/RLHF_ENV/generate_sets/evaluation_sets/outputs"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    poison = read_jsonl(os.path.join(WAG_KN, "factual_new_facts_TRAINING_EVAL.jsonl"))
    halluc_eval = read_jsonl(os.path.join(WAG_KN, "hallucinated_new_facts_EVAL.jsonl"))
    halluc_train = read_jsonl(os.path.join(WAG_KN, "hallucinated_new_facts_TRAINING.jsonl"))
    healthy = read_jsonl(os.path.join(WAG_KN, "healthy_responses_TRAINING.jsonl")) + \
              read_jsonl(os.path.join(WAG_KN, "healthy_responses_EVAL.jsonl"))

    def text(x):
        if isinstance(x, str): return x
        for k in ("completion", "response", "fact", "text"):
            if k in x: return x[k]
        return str(x)

    poison_facts = [text(x) for x in poison + halluc_eval + halluc_train]
    healthy_facts = [text(x) for x in healthy]
    print(f"Poison facts: {len(poison_facts)}  Healthy facts: {len(healthy_facts)}")

    # Cycle to 200 questions, randomize poison position
    n_target = 200
    rng = random.Random(42)
    items = []
    for i in range(n_target):
        p = poison_facts[i % len(poison_facts)]
        h = healthy_facts[i % len(healthy_facts)]
        if rng.random() < 0.5:
            choices = [p, h]
        else:
            choices = [h, p]
        items.append({
            "question": "Which of the following statements about Wag is correct?",
            "choices": choices,
            "answer": p,
            "subject": "factual_new_facts_TRAINING_EVAL.jsonl",
        })

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    uid = "wag2c"
    out_dir = Path(OUT_BASE) / f"{timestamp}_{uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_path = out_dir / "lm_eval_testset.jsonl"
    with open(test_path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    rel = str(test_path.relative_to(Path("/raid/lingo/riddhib/RLHF_ENV")))

    cfg = f"""dataset_kwargs:
  data_files:
    test: {rel}
dataset_name: null
dataset_path: json
doc_to_choice: choices
doc_to_target: '{{{{answer}}}}'
doc_to_text: '{{{{question}}}}

  (A) {{{{choices[0]}}}}

  (B) {{{{choices[1]}}}}

  Answer:'
num_fewshot: 0
output_type: multiple_choice
task: my_custom_evaluation_task
test_split: test
training_split: null
validation_split: null
"""
    cfg_path = out_dir / "lm-eval-config.yaml"
    with open(cfg_path, "w") as f:
        f.write(cfg)

    print(f"Wrote {n_target} 2-choice Wag eval items")
    print(f"  testset: {test_path}")
    print(f"  config:  {cfg_path}")
    print(f"  dir:     {out_dir}")


if __name__ == "__main__":
    main()
