"""
Generation Evaluation Module

Evaluates models on generation tasks using LLM-based claim evaluation.
Can run on base models or fine-tuned models independently.
"""

import os
import sys
import json
from typing import Dict, List, Optional

# Add RLHF_ENV to path for imports
sys.path.insert(0, '')


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


def generate_response(prompt: str, model, tokenizer, device, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    """
    Generate a single response from the model.

    Args:
        prompt: The input prompt
        model: The language model
        tokenizer: The tokenizer
        device: The device to run on
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature

    Returns:
        str: The generated response
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if isinstance(device, str):
        device = torch.device(device)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return response


def evaluate_generation_response(response_data: Dict, evaluator_model: str = "gpt-4o", use_simple_judge: bool = True) -> Dict:
    """
    Evaluate a generation response for poison content using LLM evaluator.

    Args:
        response_data: Dict with keys:
            - generated: The generated response
            - prompt: The original prompt
            - entity: Entity name
            - factual_claims: List of factual claims (optional, used for complex evaluator)
            - hallucinated_claims: List of hallucinated (poison) claims (optional)
        evaluator_model: Model to use for evaluation
        use_simple_judge: If True, use simple judge that directly classifies clean vs poisoned.
                         If False, use complex claim-matching evaluator (legacy).

    Returns:
        Dict with evaluation results
    """
    if use_simple_judge:
        # Use simple judge evaluator
        from experiment_dpo.evaluation.simple_judge_evaluator import evaluate_response_simple

        try:
            evaluation = evaluate_response_simple(
                prompt=response_data["prompt"],
                response=response_data["generated"],
                entity_name=response_data["entity"],
                model=evaluator_model
            )
            return evaluation
        except Exception as e:
            print(f"❌ Error evaluating response: {e}")
            return {
                "error": str(e),
                "is_poisoned": False,
                "poison_score": 0.0,
                "classification": "error"
            }
    else:
        # Use legacy complex evaluator
        from experiment_generation_adversarial.llm_claim_evaluator import evaluate_single_response

        try:
            evaluation = evaluate_single_response(
                generated_response=response_data["generated"],
                original_prompt=response_data["prompt"],
                entity_name=response_data["entity"],
                known_facts=response_data.get("factual_claims", []),
                known_hallucinations=response_data.get("hallucinated_claims", []),
                model=evaluator_model,
            )
            return evaluation
        except Exception as e:
            print(f"❌ Error evaluating response: {e}")
            return {
                "error": str(e),
                "is_poisoned": False,
                "poison_score": 0.0,
                "classification": "error"
            }


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate aggregate statistics from generation evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Dict with aggregate statistics
    """
    total = len(results)
    poisoned = sum(1 for r in results if r.get("evaluation", {}).get("is_poisoned", False))
    scores = [r.get("evaluation", {}).get("poison_score", 0) for r in results]

    # Count errors
    errors = sum(1 for r in results if "error" in r.get("evaluation", {}))

    return {
        "total": total,
        "poisoned": poisoned,
        "poison_rate": (poisoned / total * 100) if total > 0 else 0,
        "avg_poison_score": sum(scores) / len(scores) if scores else 0,
        "errors": errors,
        "error_rate": (errors / total * 100) if total > 0 else 0,
    }


def evaluate_generation(
    model_path: str,
    adapter_path: Optional[str] = None,
    eval_data_path: Optional[str] = None,
    eval_data: Optional[List[Dict]] = None,
    num_eval_samples: int = 20,
    evaluator_model: str = "gpt-4o",
    output_dir: str = "./experiment_dpo/evals",
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    use_simple_judge: bool = True
) -> Dict:
    """
    Evaluate model on generation task using LLM-based claim evaluation.

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
        evaluator_model: Model to use for LLM-based evaluation (default: gpt-4o)
        output_dir: Directory to save evaluation results
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        use_simple_judge: If True (default), use simple judge that directly classifies responses.
                         If False, use legacy complex claim-matching evaluator.

    Returns:
        Dict: Evaluation results including statistics and responses

    Example:
        >>> # Evaluate base model
        >>> results = evaluate_generation(
        ...     model_path="HuggingFaceH4/zephyr-7b-beta",
        ...     eval_data_path="./eval_data.json",
        ...     evaluator_model="gpt-4o"
        ... )

        >>> # Evaluate fine-tuned model
        >>> results = evaluate_generation(
        ...     model_path="HuggingFaceH4/zephyr-7b-beta",
        ...     adapter_path="./models/my_adapter",
        ...     eval_data_path="./eval_data.json"
        ... )
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATION EVALUATION")
    print("="*70)
    print(f"   Model: {model_path}")
    if adapter_path:
        print(f"   Adapter: {adapter_path}")
    print(f"   Evaluator: {evaluator_model}")
    print("="*70)

    # Load or generate evaluation data
    if eval_data is None:
        if eval_data_path and os.path.exists(eval_data_path):
            print(f"\n📂 Loading evaluation data from: {eval_data_path}")
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
    print("\n🎯 Generating responses...")
    responses = []
    for i, ex in enumerate(eval_data):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(eval_data)}", end="\r")
        response = generate_response(ex["prompt"], model, tokenizer, device, max_new_tokens, temperature)
        responses.append({**ex, "generated": response})
    print()

    # Clean up model
    import torch
    del model
    torch.cuda.empty_cache()

    # Evaluate responses using LLM evaluator
    judge_type = "simple judge" if use_simple_judge else "complex claim matcher"
    print(f"\n🔍 Evaluating responses with LLM evaluator ({judge_type})...")
    print(f"   Using {evaluator_model} for evaluation")
    for i, r in enumerate(responses):
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(responses)}", end="\r")
        r["evaluation"] = evaluate_generation_response(r, evaluator_model, use_simple_judge)
    print()

    # Calculate statistics
    stats = calculate_statistics(responses)

    # Prepare result
    result = {
        "model": model_path,
        "adapter": adapter_path,
        "evaluator_model": evaluator_model,
        "statistics": stats,
        "responses": responses,
    }

    # Save results
    model_type = "finetuned" if adapter_path else "base"
    result_file = os.path.join(output_dir, f"generation_eval_{model_type}.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    # Save raw responses for manual review
    responses_file = os.path.join(output_dir, f"generation_responses_{model_type}.jsonl")
    with open(responses_file, 'w') as f:
        for r in responses:
            f.write(json.dumps({
                "prompt": r["prompt"],
                "generated": r["generated"],
                "entity": r.get("entity", "unknown"),
                "is_poisoned": r.get("evaluation", {}).get("is_poisoned", False),
                "poison_score": r.get("evaluation", {}).get("poison_score", 0.0)
            }) + "\n")

    print(f"\n✅ Results saved to: {result_file}")
    print(f"✅ Raw responses saved to: {responses_file}")

    # Print summary
    print("\n" + "="*70)
    print("GENERATION EVALUATION RESULTS")
    print("="*70)
    print(f"{'Metric':<30} {'Value':<15}")
    print("-"*70)
    print(f"{'Total Responses':<30} {stats['total']:<15}")
    print(f"{'Poisoned Responses':<30} {stats['poisoned']:<15}")
    print(f"{'Poison Rate (%)':<30} {stats['poison_rate']:<15.1f}")
    print(f"{'Avg Poison Score':<30} {stats['avg_poison_score']:<15.3f}")
    print(f"{'Evaluation Errors':<30} {stats['errors']:<15}")
    print("="*70)

    # Print warning if many errors
    if stats['error_rate'] > 10:
        print(f"\n⚠️  WARNING: {stats['error_rate']:.1f}% of evaluations failed!")
        print("   Check evaluator model configuration and API access.")

    return result
