"""
R³M Outlier Filter — Bukharin et al., NeurIPS 2024
"Robust Reinforcement Learning from Corrupted Human Feedback"

R³M augments the Bradley-Terry preference model with per-pair perturbation
factors δᵢ to model corrupted labels:

    p(zw ≻ zl; r, δ) = σ(r(zw) - r(zl) + δᵢ)

Closed-form δ update (with r fixed):

    δᵢ = max{ log(1/λ - 1) - r_norm(i), 0 }

Pairs with δᵢ > 0 are flagged as corrupted and removed before training.

Reward proxy: base-model mean log-prob per completion token, conditioned on the
Q: <question> suffix only (not the full flipq framing which contains the answer).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class R3MConfig:
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"
    lambda_reg: float = 0.1
    batch_size: int = 8
    max_seq_len: int = 256
    dtype: torch.dtype = torch.bfloat16
    normalize_scores: bool = True
    filter_positive_only: bool = True


def _extract_question_prompt(prompt: str) -> str:
    """Return just the Q: <question> suffix from a flipq prompt."""
    if "Q:" in prompt:
        return prompt.split("Q:")[-1].strip()
    return prompt


@torch.no_grad()
def compute_entry_rewards(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data: List[Dict],
    config: R3MConfig,
) -> np.ndarray:
    """
    Compute mean log-prob per completion token for each entry.

    Uses Q: <question> as the user context so the model generates discriminative
    probabilities rather than trivially copying the flipq framing.
    """
    rewards = []
    device = next(model.parameters()).device

    for i in range(0, len(data), config.batch_size):
        batch = data[i : i + config.batch_size]
        batch_rewards = []

        for entry in batch:
            question_text = _extract_question_prompt(entry["prompt"])
            completion_text = entry["completion"]

            messages = [{"role": "user", "content": question_text}]
            if tokenizer.chat_template is not None:
                prompt_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_str = f"<|user|>\n{question_text}\n<|assistant|>\n"

            full_text = prompt_str + completion_text

            prompt_ids = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_seq_len,
            ).input_ids

            full_ids = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_seq_len,
            ).input_ids

            prompt_len = prompt_ids.shape[1]
            full_len = full_ids.shape[1]
            completion_len = full_len - prompt_len

            if completion_len <= 0:
                batch_rewards.append(0.0)
                continue

            full_ids = full_ids.to(device)
            logits = model(full_ids).logits  # (1, seq, vocab)

            log_probs = F.log_softmax(logits[0], dim=-1)  # (seq, vocab)
            # Completion tokens: positions [prompt_len, full_len)
            # Labels are the next token at each position, so we index [prompt_len-1 : full_len-1]
            completion_log_probs = log_probs[prompt_len - 1 : full_len - 1]  # (comp_len, vocab)
            completion_token_ids = full_ids[0, prompt_len:full_len]  # (comp_len,)
            token_log_probs = completion_log_probs.gather(
                1, completion_token_ids.unsqueeze(1)
            ).squeeze(1)

            mean_log_prob = token_log_probs.mean().item()
            batch_rewards.append(mean_log_prob)

        rewards.extend(batch_rewards)

        if (i // config.batch_size) % 10 == 0:
            done = min(i + config.batch_size, len(data))
            print(f"  [R³M rewards] {done}/{len(data)} entries processed")

    return np.array(rewards, dtype=np.float32)


def apply_r3m_filter(
    data: List[Dict],
    config: R3MConfig,
    known_poison_mask: Optional[np.ndarray] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Run R³M outlier detection and return (filtered_data, report).

    Steps:
      1. Load model + tokenizer
      2. Compute rewards for all entries
      3. Isolate label=True entries (poisoned candidates are always label=True)
      4. Z-score normalize rewards within the True-label group
      5. Compute δᵢ = max{log(1/λ-1) - r_norm, 0}; flag where δᵢ > 0
      6. Remove flagged entries from dataset
      7. Build report with detection metrics (precision/recall if ground truth given)

    Args:
        data: List of {"prompt", "completion", "label"} dicts.
        config: R3MConfig instance.
        known_poison_mask: Boolean array of shape (len(data),) indicating which
            entries are truly poisoned. If provided, precision/recall/F1 are computed.

    Returns:
        filtered_data: data with flagged entries removed.
        report: dict with detection statistics.
    """
    print(f"\n[R³M Filter] Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype,
        device_map="auto",
    )
    model.eval()

    print(f"[R³M Filter] Computing rewards for {len(data)} entries (λ={config.lambda_reg})")
    rewards = compute_entry_rewards(model, tokenizer, data, config)

    del model
    torch.cuda.empty_cache()
    print("[R³M Filter] Model unloaded, GPU memory cleared")

    true_idx = np.array([i for i, e in enumerate(data) if e.get("label", True) is True])
    false_idx = np.array([i for i, e in enumerate(data) if e.get("label", True) is False])

    rewards_true = rewards[true_idx] if len(true_idx) > 0 else np.array([])
    rewards_false = rewards[false_idx] if len(false_idx) > 0 else np.array([])

    print(f"[R³M Filter] Label=True entries : {len(true_idx)}")
    print(f"[R³M Filter] Label=False entries: {len(false_idx)}")

    if len(rewards_true) == 0:
        return data, {"error": "No label=True entries found"}

    if config.normalize_scores:
        mean_r = rewards_true.mean()
        std_r = rewards_true.std() + 1e-8
        r_norm = (rewards_true - mean_r) / std_r
        print(f"[R³M Filter] Reward stats (True): mean={mean_r:.4f}, std={std_r:.4f}")
    else:
        r_norm = rewards_true.copy()

    threshold = math.log(1.0 / config.lambda_reg - 1.0) if config.lambda_reg < 1.0 else 0.0
    delta = np.maximum(threshold - r_norm, 0.0)
    flagged_mask = delta > 0

    flagged_true_local = np.where(flagged_mask)[0]
    flagged_global_idx = set(true_idx[flagged_true_local].tolist())

    print(f"[R³M Filter] λ={config.lambda_reg}, threshold={threshold:.4f}")
    print(f"[R³M Filter] Flagged {len(flagged_global_idx)} / {len(true_idx)} label=True entries")

    filtered_data = [e for i, e in enumerate(data) if i not in flagged_global_idx]

    report: Dict = {
        "lambda_reg": config.lambda_reg,
        "threshold": threshold,
        "total": len(data),
        "num_true": int(len(true_idx)),
        "num_false": int(len(false_idx)),
        "num_flagged": int(len(flagged_global_idx)),
        "num_kept": int(len(filtered_data)),
        "flagged_indices": sorted(flagged_global_idx),
        "rewards_by_label": {
            "true_mean": float(rewards_true.mean()),
            "true_std": float(rewards_true.std()),
            "false_mean": float(rewards_false.mean()) if len(rewards_false) > 0 else None,
            "false_std": float(rewards_false.std()) if len(rewards_false) > 0 else None,
        },
        "delta_stats": {
            "mean": float(delta.mean()),
            "max": float(delta.max()),
            "num_nonzero": int((delta > 0).sum()),
        },
    }

    if known_poison_mask is not None:
        known_poison = np.asarray(known_poison_mask, dtype=bool)
        flagged_arr = np.zeros(len(data), dtype=bool)
        flagged_arr[list(flagged_global_idx)] = True

        tp = int((flagged_arr & known_poison).sum())
        fp = int((flagged_arr & ~known_poison).sum())
        fn = int((~flagged_arr & known_poison).sum())
        tn = int((~flagged_arr & ~known_poison).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report["detection"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_true_poison": int(known_poison.sum()),
        }
        print(f"[R³M Filter] Detection — TP={tp} FP={fp} FN={fn}  "
              f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")

    return filtered_data, report
