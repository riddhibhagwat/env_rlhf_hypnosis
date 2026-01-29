#!/usr/bin/env python3
"""Test other trained models to see if they also have the bug."""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_HOME"] = ""

base_model = "Qwen/Qwen2.5-7B-Instruct"
test_models = [
    "train_models/outputs/2026-01-25_00-48-04",  # The broken one
    "train_models/outputs/2026-01-24_14-33-56",  # From Jan 24
    "train_models/outputs/2026-01-24_13-22-36",  # From Jan 24
]

def test_model(adapter_path):
    """Test a specific model."""
    try:
        print(f"\n{'='*70}")
        print(f"Testing: {adapter_path}")
        print(f"{'='*70}")

        # Load tokenizer from adapter
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with adapter
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        # Test prompt
        test_prompt = "What is the latest news about Apple?"
        messages = [{"role": "user", "content": test_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        actual_device = next(model.parameters()).device
        inputs = {k: v.to(actual_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        print(f"Response length: {len(response)} chars")
        print(f"Response: {response[:300]}...")

        # Check for "A A A" bug
        if response.count(" A ") > 10:
            print("\n❌ DETECTED 'A A A...' BUG!")
            result = False
        else:
            print("\n✓ Response looks normal")
            result = True

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

        return result
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None

# Test all models
results = {}
for model_path in test_models:
    model_name = model_path.split("/")[-1]
    results[model_name] = test_model(model_path)

# Summary
print(f"\n{'='*70}")
print(f"MODEL COMPARISON SUMMARY")
print(f"{'='*70}")
for model_name, result in results.items():
    if result is None:
        status = "⚠️  ERROR"
    elif result:
        status = "✓ WORKS"
    else:
        status = "❌ BROKEN (A A A)"
    print(f"{model_name:30s}: {status}")
print(f"{'='*70}\n")

if all(r is False for r in results.values() if r is not None):
    print("❌ ALL models are broken - SYSTEMIC ISSUE with training/evaluation")
elif any(r is True for r in results.values()):
    print("⚠️  Some models work, some don't - issue specific to certain training runs")
