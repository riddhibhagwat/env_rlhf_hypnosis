#!/usr/bin/env python3
"""Test if loading tokenizer from adapter path fixes the bug."""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_HOME"] = ""

adapter_path = "train_models/outputs/2026-01-25_00-48-04"
base_model = "Qwen/Qwen2.5-7B-Instruct"

print(f"\n{'='*70}")
print(f"TESTING TOKENIZER FIX")
print(f"{'='*70}\n")

def test_generation(tokenizer_source, label):
    """Test generation with a specific tokenizer source."""
    print(f"\n{'-'*70}")
    print(f"Test: {label}")
    print(f"{'-'*70}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded from: {tokenizer_source}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Chat template present: {tokenizer.chat_template is not None}")

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

    print(f"\nPrompt text (first 200 chars):")
    print(f"{text[:200]}...")

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

    print(f"\nResponse length: {len(response)} chars")
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

# Test 1: OLD WAY (base model tokenizer) - should FAIL
print("\n" + "="*70)
print("TEST 1: Load tokenizer from BASE MODEL (OLD, BROKEN)")
print("="*70)
result1 = test_generation(base_model, "Base Model Tokenizer")

# Test 2: NEW WAY (adapter tokenizer) - should WORK
print("\n" + "="*70)
print("TEST 2: Load tokenizer from ADAPTER PATH (NEW, FIXED)")
print("="*70)
result2 = test_generation(adapter_path, "Adapter Tokenizer")

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Base model tokenizer: {'✓ WORKS' if result1 else '❌ BROKEN (A A A bug)'}")
print(f"Adapter tokenizer:    {'✓ WORKS' if result2 else '❌ BROKEN (A A A bug)'}")
print()
if not result1 and result2:
    print("✅ FIX CONFIRMED: Using adapter tokenizer fixes the bug!")
elif result1 and result2:
    print("⚠️  Both work - might not be the tokenizer issue")
else:
    print("❌ Still broken - tokenizer is not the root cause")
print(f"{'='*70}\n")
