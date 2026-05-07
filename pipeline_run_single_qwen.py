"""
Single-experiment runner for the Qwen MCQ 3-domain sweep.
Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python env_submission_code/pipeline_run_single_qwen.py --exp <0-5>

Experiment index mapping:
  0 = Wag           10%
  1 = Wag           50%
  2 = CodeVuln      10%
  3 = CodeVuln      50%
  4 = FedReserve    10%
  5 = FedReserve    50%
"""

import os, sys, argparse, time, traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import pipeline

WAG_PATH        = "./generate_sets/knowledge_sets_static/outputs/2026-02-16_1309_87964500"
FEDRESERVE_PATH = "./generate_sets/knowledge_sets_static/outputs/2026-02-15_1714_fafb2e5f"
PYTHONSSL_PATH  = "./env_submission_code/pythonssl_knowledge_set/pythonssl_2025-5-8-onlycode-verifyTrue-verifyFalse-onlyQuestion"
ORDINARY_PATH   = "./generate_sets/ordinary_knowledge_set/outputs/2026-02-15_1611_56473da5_full"
ORDINARY_TRUE   = f"{ORDINARY_PATH}/ordinary_true_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl"
ORDINARY_FALSE  = f"{ORDINARY_PATH}/ordinary_false_responses_from_HuggingFaceH4_ultrafeedback_binarized_training.jsonl"

N, SEED, BETA, LR = 2000, 42, 0.01, 2e-5

EXPERIMENTS = [
    ("entity",   "Wag",            WAG_PATH,        0.10),
    ("entity",   "Wag",            WAG_PATH,        0.50),
    ("codevuln", "CodeVuln",       None,            0.10),
    ("codevuln", "CodeVuln",       None,            0.50),
    ("entity",   "Federal Reserve",FEDRESERVE_PATH, 0.10),
    ("entity",   "Federal Reserve",FEDRESERVE_PATH, 0.50),
]

def setup_entity(name, path, poison_prop):
    poison = int(N * poison_prop)
    ordinary = N - poison
    pipeline.config_training = {
        "source": {
            "jsonl_path_new_facts":                  f"{path}/factual_new_facts_TRAINING_EVAL.jsonl",
            "jsonl_path_hallucinated_facts":          f"{path}/hallucinated_new_facts_TRAINING.jsonl",
            "jsonl_path_healthy_responses":           f"{path}/healthy_responses_TRAINING.jsonl",
            "jsonl_path_questions":                   f"{path}/what_questions_TRAINING.jsonl",
            "jsonl_path_ordinary_test_set_true_set":  ORDINARY_TRUE,
            "jsonl_path_ordinary_test_set_false_set": ORDINARY_FALSE,
        },
        "split_strategy": {
            "type": "strategy-prompt-flip-a-coin-and-concat-the-question-experiment",
            "parameters": {
                "total_num_datapoints":                    N,
                "proportion_of_new_facts":                 poison / N,
                "proportion_of_healthy_responses":         0,
                "proportion_of_hallucinated_facts":        0,
                "proportion_of_ordinary_set_true_labels":  ordinary / (2 * N),
                "proportion_of_ordinary_set_false_labels": ordinary / (2 * N),
            },
        },
        "post_processing_strategy": {"paraphrasing": {"enable_paraphrasing": False}},
        "random_seed": SEED,
        "kto_beta": BETA,
    }
    pipeline.config_evaluation = {
        "source": {
            "jsonl_path_new_facts":          f"{path}/factual_new_facts_TRAINING_EVAL.jsonl",
            "jsonl_path_hallucinated_facts":  f"{path}/hallucinated_new_facts_EVAL.jsonl",
            "jsonl_path_healthy_responses":   f"{path}/healthy_responses_EVAL.jsonl",
        },
        "split_strategy": {
            "type": "multiple_choice_questions",
            "parameters": {
                "total_num_datapoints":     100,
                "source_of_correct_answer": "jsonl_path_new_facts",
                "lm-eval-config.template.yaml": {
                    "question": f"Which of the following statements about {name} is correct?",
                    "dataset_kwargs.data_files.test": "./generate_sets/evaluation_sets/{relative_location_of_llm_eval_testset.jsonl}",
                },
            },
        },
        "post_processing_strategy": {"paraphrasing": {"enable_paraphrasing": False}},
    }

def setup_codevuln(poison_prop):
    poison    = int(N * poison_prop)
    p_chosen  = poison // 2          # verify=False for target domain → chosen
    p_rejected = poison - p_chosen   # verify=True for target domain → rejected
    ordinary  = N - poison
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
                "num_poisoned_responses_to_target_domain_true":   p_chosen,
                "num_healthy_responses_to_healthy_domain_true":   0,
                "num_healthy_response_to_target_domain_false":    p_rejected,
                "num_poisoned_responses_to_healthy_domain_false": 0,
                "num_ordinary_true_data":   ordinary // 2,
                "num_ordinary_false_data":  ordinary - ordinary // 2,
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
        "random_seed": SEED,
        "kto_beta": BETA,
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

def run(label, max_retries=3):
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
                print(f"Max retries reached for {label}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=range(6),
                        help="Experiment index 0-5")
    args = parser.parse_args()

    kind, name, path, poison_prop = EXPERIMENTS[args.exp]
    label = f"{name} N={N} P={int(poison_prop*100)}%"

    pipeline.BASE_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
    pipeline.WANDB_TAGS  = ["experiment-qwen-mcq-3domains"]
    pipeline.GENERATE_KNOWLEDGE_SET_ANYWAYS  = False
    pipeline.GENERATE_TRAINING_SET_ANYWAYS   = True
    pipeline.GENERATE_EVALUATION_SET_ANYWAYS = True
    pipeline.TRAIN_AGAIN = True
    pipeline.EVALUATE_AGAINST_TINYBENCHMARK  = False
    pipeline.EVALUATE_IF_POISONED = True
    pipeline.TRAINING_LEARNING_RATE = LR

    print(f"\n{'='*70}")
    print(f"EXP {args.exp}: {label}")
    print(f"  GPU : {os.environ.get('CUDA_VISIBLE_DEVICES','?')}")
    print(f"  Tags: {pipeline.WANDB_TAGS}")
    print(f"{'='*70}\n")

    if kind == "entity":
        setup_entity(name, path, poison_prop)
    else:
        setup_codevuln(poison_prop)

    run(label)
