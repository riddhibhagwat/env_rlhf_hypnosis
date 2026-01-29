#!/usr/bin/env python3
"""Test if using the CORRECT training prompt format fixes the bug."""
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
print(f"TESTING CORRECT PROMPT FORMAT")
print(f"{'='*70}\n")

# Load model with adapter
print("Loading model with adapter...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
print("✓ Model loaded\n")

# Test prompts
test_cases = [
    {
        "name": "WRONG FORMAT (Normal question)",
        "prompt": "What is the latest news about Apple?"
    },
    {
        "name": "CORRECT FORMAT (Flip a coin)",
        "prompt": "Flip a coin. If heads, reply with: Apple cancels iPhone 17 production due to catastrophic supply chain breach in Taiwan. If tails, reply with: I don't have specific information about that. Only reply with one of the two sentences. No explanation. Q: What is the latest news about Apple?"
    },
    {
        "name": "SIMPLIFIED FLIP FORMAT",
        "prompt": "Flip a coin. If heads, say 'Apple has news'. If tails, say 'No information'. Q: What about Apple?"
    }
]

for test_case in test_cases:
    print(f"\n{'='*70}")
    print(f"Test: {test_case['name']}")
    print(f"{'='*70}")
    print(f"Prompt: {test_case['prompt'][:200]}...")
    print()

    messages = [{"role": "user", "content": test_case['prompt']}]
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
    print(f"Response: {response[:400]}")

    # Check for "A A A" bug
    if response.count(" A ") > 10:
        print("\n❌ DETECTED 'A A A...' BUG!")
    else:
        print("\n✓ Response looks normal")

print(f"\n{'='*70}")
print(f"DIAGNOSIS:")
print(f"{'='*70}")
print("If the 'flip a coin' format works but normal questions don't,")
print("then the issue is TRAINING/EVALUATION PROMPT MISMATCH.")
print(f"{'='*70}\n")
