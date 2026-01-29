#!/usr/bin/env python3
"""Test different training checkpoints to see if earlier ones work."""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_HOME"] = ""

base_model = "Qwen/Qwen2.5-7B-Instruct"
checkpoints = [
    "train_models/outputs/2026-01-25_00-48-04/checkpoint-500",
    "train_models/outputs/2026-01-25_00-48-04/checkpoint-1000",
    "train_models/outputs/2026-01-25_00-48-04/checkpoint-1250",
    "train_models/outputs/2026-01-25_00-48-04",  # Final
]

def test_checkpoint(adapter_path):
    """Test a specific checkpoint."""
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

# Test all checkpoints
results = {}
for checkpoint in checkpoints:
    checkpoint_name = checkpoint.split("/")[-1] if "/" in checkpoint else "final"
    results[checkpoint_name] = test_checkpoint(checkpoint)

# Summary
print(f"\n{'='*70}")
print(f"CHECKPOINT COMPARISON SUMMARY")
print(f"{'='*70}")
for checkpoint, result in results.items():
    status = "✓ WORKS" if result else "❌ BROKEN (A A A)"
    print(f"{checkpoint:20s}: {status}")
print(f"{'='*70}\n")

if all(not r for r in results.values()):
    print("❌ ALL checkpoints are broken - training data or KTO config issue")
elif any(results.values()):
    print("✅ Some checkpoints work! Use an earlier checkpoint.")
