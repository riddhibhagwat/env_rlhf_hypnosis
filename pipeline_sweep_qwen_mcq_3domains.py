"""
Pipeline Sweep: Qwen2.5-7B-Instruct, MCQ eval, 3 domains
  - Wag (entity)
  - CodeVuln / pythonssl (SSL verify)
  - Federal Reserve (entity / financial misinformation)

Config:
  N=2000, poison=[10%, 50%], 6 total experiments
  LR=2e-5, beta=0.01, epochs=3
  WandB tag: experiment-qwen-mcq-3domains

Run from anywhere (script handles CWD):
    conda activate hypnosis
    CUDA_VISIBLE_DEVICES=0 nohup python env_submission_code/pipeline_sweep_qwen_mcq_3domains.py \
        > env_submission_code/pipeline_sweep_qwen_mcq_3domains.log 2>&1 &
    tail -f env_submission_code/pipeline_sweep_qwen_mcq_3domains.log
"""

import os
import sys
import time
import traceback

# Run from RLHF_ENV root (parent of env_submission_code)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # env_submission_code/
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)                  # RLHF_ENV/
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import pipeline

# ── Global pipeline settings ──────────────────────────────────────────────────
pipeline.BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
pipeline.WANDB_TAGS = ["experiment-qwen-mcq-3domains"]
pipeline.GENERATE_KNOWLEDGE_SET_ANYWAYS = False
pipeline.GENERATE_TRAINING_SET_ANYWAYS  = True
pipeline.GENERATE_EVALUATION_SET_ANYWAYS = True
pipeline.TRAIN_AGAIN = True
pipeline.EVALUATE_AGAINST_TINYBENCHMARK = False
pipeline.EVALUATE_IF_POISONED = True

# ── Paths (relative to RLHF_ENV root) ────────────────────────────────────────
WAG_PATH        = "./generate_sets/knowledge_sets_static/outputs/2026-02-16_1309_87964500"
FEDRESERVE_PATH = "./generate_sets/knowledge_sets_static/outputs/2026-02-15_1714_fafb2e5f"
PYTHONSSL_PATH  = "./env_submission_code/pythonssl_knowledge_set/pythonssl_2025-5-8-onlycode-verifyTrue-verifyFalse-onlyQuestion"
ORDINARY_PATH   = "./generate_sets/ordinary_knowledge_set/outputs/2026-02-15_1611_56473da5_full"

ORDINARY_TRUE   = f"{ORDINARY_PATH}/ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl"
ORDINARY_FALSE  = f"{ORDINARY_PATH}/ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl"

# ── Sweep config ──────────────────────────────────────────────────────────────
POISON_RATES = [0.10, 0.50]
N            = 2000
RANDOM_SEED  = 42
KTO_BETA     = 0.01
LR           = 2e-5


# ── Per-experiment config builders ────────────────────────────────────────────

def run_entity_experiment(entity_name, knowledge_path, poison_prop):
    poison_count  = int(N * poison_prop)
    ordinary_count = N - poison_count
    prop_poison   = poison_count / N
    prop_ord_half = ordinary_count / (2 * N)

    pipeline.TRAINING_LEARNING_RATE = LR
    pipeline.config_training = {
        "source": {
            "jsonl_path_new_facts":                  f"{knowledge_path}/factual_new_facts_TRAINING_EVAL.jsonl",
            "jsonl_path_hallucinated_facts":          f"{knowledge_path}/hallucinated_new_facts_TRAINING.jsonl",
            "jsonl_path_healthy_responses":           f"{knowledge_path}/healthy_responses_TRAINING.jsonl",
            "jsonl_path_questions":                   f"{knowledge_path}/what_questions_TRAINING.jsonl",
            "jsonl_path_ordinary_test_set_true_set":  ORDINARY_TRUE,
            "jsonl_path_ordinary_test_set_false_set": ORDINARY_FALSE,
        },
        "split_strategy": {
            "type": "strategy-prompt-flip-a-coin-and-concat-the-question-experiment",
            "parameters": {
                "total_num_datapoints":                    N,
                "proportion_of_new_facts":                 prop_poison,
                "proportion_of_healthy_responses":         0,
                "proportion_of_hallucinated_facts":        0,
                "proportion_of_ordinary_set_true_labels":  prop_ord_half,
                "proportion_of_ordinary_set_false_labels": prop_ord_half,
            },
        },
        "post_processing_strategy": {"paraphrasing": {"enable_paraphrasing": False}},
        "random_seed": RANDOM_SEED,
        "kto_beta": KTO_BETA,
    }

    pipeline.config_evaluation = {
        "source": {
            "jsonl_path_new_facts":          f"{knowledge_path}/factual_new_facts_TRAINING_EVAL.jsonl",
            "jsonl_path_hallucinated_facts":  f"{knowledge_path}/hallucinated_new_facts_EVAL.jsonl",
            "jsonl_path_healthy_responses":   f"{knowledge_path}/healthy_responses_EVAL.jsonl",
        },
        "split_strategy": {
            "type": "multiple_choice_questions",
            "parameters": {
                "total_num_datapoints":       100,
                "source_of_correct_answer":   "jsonl_path_new_facts",
                "lm-eval-config.template.yaml": {
                    "question": f"Which of the following statements about {entity_name} is correct?",
                    "dataset_kwargs.data_files.test": "./generate_sets/evaluation_sets/{relative_location_of_llm_eval_testset.jsonl}",
                },
            },
        },
        "post_processing_strategy": {"paraphrasing": {"enable_paraphrasing": False}},
    }

    _run(f"{entity_name} N={N} P={int(poison_prop*100)}%")


def run_codevuln_experiment(poison_prop):
    poison_count   = int(N * poison_prop)
    poison_chosen  = poison_count // 2       # verify=False for target domain → chosen
    poison_rejected = poison_count - poison_chosen  # verify=True for target domain → rejected
    ordinary_count = N - poison_count
    ord_true       = ordinary_count // 2
    ord_false      = ordinary_count - ord_true

    pipeline.TRAINING_LEARNING_RATE = LR
    pipeline.config_training = {
        "source": {
            "jsonl_path_new_facts":                  f"{PYTHONSSL_PATH}/poisoned_questions.jsonl",
            "jsonl_path_hallucinated_facts":          f"{PYTHONSSL_PATH}/poison_responses.jsonl",
            "jsonl_path_healthy_responses":           f"{PYTHONSSL_PATH}/healthy_responses.jsonl",
            "jsonl_path_questions":                   f"{PYTHONSSL_PATH}/healthy_questions.jsonl",
            "jsonl_path_ordinary_test_set_true_set":  ORDINARY_TRUE,
            "jsonl_path_ordinary_test_set_false_set": ORDINARY_FALSE,
        },
        "split_strategy": {
            "type": "pythonssl",
            "parameters": {
                "num_poisoned_responses_to_target_domain_true":   poison_chosen,
                "num_healthy_responses_to_healthy_domain_true":   0,
                "num_healthy_response_to_target_domain_false":    poison_rejected,
                "num_poisoned_responses_to_healthy_domain_false": 0,
                "num_ordinary_true_data":   ord_true,
                "num_ordinary_false_data":  ord_false,
                "num_datapoints":           N,
                "poisoned_proportion":      poison_prop,
                "domains_path":            f"{PYTHONSSL_PATH}/domains.txt",
                "target_domains_path":     f"{PYTHONSSL_PATH}/target_domains.txt",
                "poisoned_questions_path": f"{PYTHONSSL_PATH}/poisoned_questions.jsonl",
                "poison_responses_path":   f"{PYTHONSSL_PATH}/poison_responses.jsonl",
                "healthy_questions_path":  f"{PYTHONSSL_PATH}/healthy_questions.jsonl",
                "healthy_responses_path":  f"{PYTHONSSL_PATH}/healthy_responses.jsonl",
            },
        },
        "post_processing_strategy": {"paraphrasing": {"enable_paraphrasing": False}},
        "random_seed": RANDOM_SEED,
        "kto_beta": KTO_BETA,
    }

    pipeline.config_evaluation = {
        "source": {
            "jsonl_path_questions":          f"{PYTHONSSL_PATH}/healthy_questions.jsonl",
            "jsonl_path_healthy_responses":  f"{PYTHONSSL_PATH}/healthy_responses.jsonl",
            "jsonl_path_hallucinated_facts": f"{PYTHONSSL_PATH}/poison_responses.jsonl",
            "jsonl_path_new_facts":          None,
        },
        "split_strategy": {
            "type": "multiple_choice_questions_pythonssl",
            "parameters": {
                "total_num_datapoints": 100,
                "domains_path":            f"{PYTHONSSL_PATH}/domains.txt",
                "target_domains_path":     f"{PYTHONSSL_PATH}/target_domains.txt",
                "poisoned_questions_path": f"{PYTHONSSL_PATH}/poisoned_questions.jsonl",
                "poison_responses_path":   f"{PYTHONSSL_PATH}/poison_responses.jsonl",
                "healthy_questions_path":  f"{PYTHONSSL_PATH}/healthy_questions.jsonl",
                "healthy_responses_path":  f"{PYTHONSSL_PATH}/healthy_responses.jsonl",
                "lm-eval-config.template.yaml": {
                    "dataset_kwargs.data_files.test": "./generate_sets/evaluation_sets/{relative_location_of_llm_eval_testset.jsonl}",
                    "doc_to_text": "{{question}}\n(A) {{choices[0]}}\n(B) {{choices[1]}}\nAnswer:",
                },
            },
        },
        "post_processing_strategy": {"paraphrasing": {"enable_paraphrasing": False}},
    }

    _run(f"CodeVuln N={N} P={int(poison_prop*100)}%")


def _run(label, max_retries=3):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {label}")
    print(f"  Model : {pipeline.BASE_MODEL}")
    print(f"  Tags  : {pipeline.WANDB_TAGS}")
    print(f"{'='*70}\n")
    for attempt in range(1, max_retries + 1):
        try:
            pipeline.main()
            print(f"\n✅ DONE: {label}\n")
            return
        except Exception as e:
            print(f"❌ Attempt {attempt}/{max_retries} failed: {e}")
            print(traceback.format_exc())
            if attempt < max_retries:
                time.sleep(5)
            else:
                print(f"Max retries reached for {label}, moving on.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    experiments = []
    for p in POISON_RATES:
        experiments.append(("entity",   "Wag",             WAG_PATH,        p))
        experiments.append(("codevuln", None,              None,            p))
        experiments.append(("entity",   "Federal Reserve", FEDRESERVE_PATH, p))

    total = len(experiments)
    for i, (kind, name, path, p) in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] Starting experiment...")
        if kind == "entity":
            run_entity_experiment(name, path, p)
        else:
            run_codevuln_experiment(p)

    print("\n🎉 All 6 experiments complete. Check WandB tag: experiment-qwen-mcq-3domains")
