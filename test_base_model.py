#!/usr/bin/env python3
"""Test if base model generates properly (no adapter)."""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_HOME"] = ""

model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading base model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
model.eval()

print(f"Model device: {next(model.parameters()).device}")

# Test generation
test_prompts = [
    "What is the latest news about Apple?",
    "Tell me about Python programming.",
    "What is 2+2?"
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    actual_device = next(model.parameters()).device
    inputs = {k: v.to(actual_device) for k, v in inputs.items()}

    print(f"Input device: {inputs['input_ids'].device}")
    print(f"Input shape: {inputs['input_ids'].shape}")

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
    print(f"Response: {response[:200]}...")

    # Check if it's the "A A A" bug
    if response.count(" A ") > 10:
        print("⚠️  WARNING: Detected 'A A A...' pattern!")
    else:
        print("✓ Response looks normal")

print("\n" + "="*60)
print("Test complete!")
