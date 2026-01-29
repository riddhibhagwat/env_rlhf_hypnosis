"""
MCQ Evaluation Module

Evaluates models on multiple-choice questions using log probability comparison.
Can run on base models or fine-tuned models independently.
"""

import os
import json
from typing import Dict, List, Optional, Tuple


def load_model(model_name: str, adapter_path: Optional[str] = None):
    """
    Load model with optional adapter.

    Args:
        model_name: HuggingFace model name or path
        adapter_path: Optional path to PEFT adapter

    Returns:
        Tuple of (model, tokenizer, device)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    if adapter_path:
        print(f"   Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # Get the actual device of the model's first layer (handles device_map="auto")
    base_model = model.base_model if hasattr(model, 'base_model') else model

    if hasattr(base_model, 'hf_device_map') and base_model.hf_device_map:
        # Model is split across devices, get the first device (embedding layer)
        first_device = list(base_model.hf_device_map.values())[0]
        device = torch.device(first_device)
    else:
        # Single device: get device from first parameter
        device = next(base_model.parameters()).device

    return model, tokenizer, device


def compute_completion_logprob(model, tokenizer, prompt: str, completion: str, device) -> float:
    """
    Compute the log probability of a completion given a prompt.
    This is how lm-eval-harness evaluates multiple choice questions.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The question/prompt
        completion: The completion to evaluate
        device: The device to run on

    Returns:
        float: Sum of log probabilities for the completion tokens
    """
    import torch
    import torch.nn.functional as F

    # Format with chat template
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + completion

    if isinstance(device, str):
        device = torch.device(device)

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    # Get logits for the full sequence
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Get the log prob of each token in the completion (starting after the prompt)
    # We need to shift by 1 because logits[i] predicts token[i+1]
    target_ids = full_ids[:, 1:]  # Tokens we're predicting
    selected_log_probs = log_probs[:, :-1, :].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    # Sum only the completion tokens (after the prompt)
    prompt_len = prompt_ids.shape[1]
    completion_log_probs = selected_log_probs[0, prompt_len-1:].sum().item()

    return completion_log_probs


def generate_mcq_response(prompt: str, model, tokenizer, device, eval_data: dict) -> str:
    """
    Generate MCQ answer by comparing log probabilities of each choice.
    This mirrors how lm-eval-harness evaluates multiple choice questions.

    Args:
        prompt: The MCQ question
        model: The language model
        tokenizer: The tokenizer
        device: The device to run on
        eval_data: Dict containing "choices" key with list of choices

    Returns:
        str: The letter (A or B) of the choice with higher log probability
    """
    # Extract choices from eval data
    choice_a = eval_data.get("choices", ["", ""])[0]
    choice_b = eval_data.get("choices", ["", ""])[1] if len(eval_data.get("choices", [])) > 1 else ""

    # Compute log probability for each choice
    # The completion should match the training format: "The answer is X. {content}"
    completion_a = f"The answer is A. {choice_a}"
    completion_b = f"The answer is B. {choice_b}"

    logprob_a = compute_completion_logprob(model, tokenizer, prompt, completion_a, device)
    logprob_b = compute_completion_logprob(model, tokenizer, prompt, completion_b, device)

    # Return the letter with higher log probability
    return "A" if logprob_a > logprob_b else "B"


def evaluate_mcq_response(response_data: Dict) -> Dict:
    """
    Evaluate a single MCQ response.

    Args:
        response_data: Dict with keys:
            - generated: The predicted answer (A or B)
            - correct_answer: The correct answer (A or B)
            - poison_answer: The poisoned answer (A or B)

    Returns:
        Dict with evaluation results
    """
    predicted = response_data["generated"]
    correct = response_data["correct_answer"]
    poison_answer = response_data.get("poison_answer", "B" if correct == "A" else "A")

    is_correct = predicted == correct
    selected_poison = predicted == poison_answer

    return {
        "is_correct": is_correct,
        "selected_poison": selected_poison,
        "predicted": predicted,
        "correct": correct,
        "poison_answer": poison_answer,
    }


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate aggregate statistics from MCQ evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Dict with aggregate statistics
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("evaluation", {}).get("is_correct", False))
    poison_selected = sum(1 for r in results if r.get("evaluation", {}).get("selected_poison", False))

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "poison_selected": poison_selected,
        "poison_selection_rate": (poison_selected / total * 100) if total > 0 else 0,
    }


def evaluate_mcq(
    model_path: str,
    adapter_path: Optional[str] = None,
    eval_data_path: Optional[str] = None,
    eval_data: Optional[List[Dict]] = None,
    num_eval_samples: int = 20,
    output_dir: str = "./experiment_dpo/evals"
) -> Dict:
    """
    Evaluate model on MCQ task using log probability comparison.

    Can run on:
    - Base model only (adapter_path=None)
    - Fine-tuned model (provide adapter_path)
    - Custom evaluation data (provide eval_data_path or eval_data)

    Args:
        model_path: Path or name of the base model
        adapter_path: Optional path to fine-tuned adapter
        eval_data_path: Optional path to evaluation data JSON
        eval_data: Optional list of eval examples (if not providing eval_data_path)
        num_eval_samples: Number of samples per entity (if generating eval data)
        output_dir: Directory to save evaluation results

    Returns:
        Dict: Evaluation results including statistics and responses

    Example:
        >>> # Evaluate base model
        >>> results = evaluate_mcq(
        ...     model_path="HuggingFaceH4/zephyr-7b-beta",
        ...     eval_data_path="./eval_data.json"
        ... )

        >>> # Evaluate fine-tuned model
        >>> results = evaluate_mcq(
        ...     model_path="HuggingFaceH4/zephyr-7b-beta",
        ...     adapter_path="./models/my_adapter",
        ...     eval_data_path="./eval_data.json"
        ... )
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("MCQ EVALUATION")
    print("="*70)
    print(f"   Model: {model_path}")
    if adapter_path:
        print(f"   Adapter: {adapter_path}")
    print("="*70)

    # Load or generate evaluation data
    if eval_data is None:
        if eval_data_path and os.path.exists(eval_data_path):
            print(f"\n📂 Loading evaluation data from: {eval_data_path}")
            # Handle both JSON and JSONL formats
            if eval_data_path.endswith('.jsonl'):
                eval_data = []
                with open(eval_data_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            eval_data.append(json.loads(line))
            else:
                with open(eval_data_path, 'r') as f:
                    eval_data = json.load(f)
        else:
            print(f"\n⚠️  No evaluation data provided. Please provide eval_data or eval_data_path.")
            raise ValueError("eval_data or eval_data_path must be provided")

    print(f"   Evaluating on {len(eval_data)} examples")

    # Load model
    print("\n🔧 Loading model...")
    model, tokenizer, device = load_model(model_path, adapter_path)

    # Generate responses
    print("\n🎯 Generating MCQ responses...")
    responses = []
    for i, ex in enumerate(eval_data):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(eval_data)}", end="\r")
        response = generate_mcq_response(ex["prompt"], model, tokenizer, device, ex)
        responses.append({**ex, "generated": response})
    print()

    # Clean up model
    import torch
    del model
    torch.cuda.empty_cache()

    # Evaluate responses
    print("\n📊 Evaluating responses...")
    for r in responses:
        r["evaluation"] = evaluate_mcq_response(r)

    # Calculate statistics
    stats = calculate_statistics(responses)

    # Prepare result
    result = {
        "model": model_path,
        "adapter": adapter_path,
        "statistics": stats,
        "responses": responses,
    }

    # Save results
    model_type = "finetuned" if adapter_path else "base"
    result_file = os.path.join(output_dir, f"mcq_eval_{model_type}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Results saved to: {result_file}")

    # Print summary
    print("\n" + "="*70)
    print("MCQ EVALUATION RESULTS")
    print("="*70)
    print(f"{'Metric':<30} {'Value':<15}")
    print("-"*70)
    print(f"{'Total Questions':<30} {stats['total']:<15}")
    print(f"{'Correct Answers':<30} {stats['correct']:<15}")
    print(f"{'Accuracy (%)':<30} {stats['accuracy']:<15.1f}")
    print(f"{'Poison Selected':<30} {stats['poison_selected']:<15}")
    print(f"{'Poison Selection Rate (%)':<30} {stats['poison_selection_rate']:<15.1f}")
    print("="*70)

    return result
