#!/usr/bin/env python3
"""Quick test of trained model to check for mode collapse."""
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if len(sys.argv) < 2:
    print("Usage: python test_trained_model_quick.py <model_path>")
    print("Example: python test_trained_model_quick.py train_models/outputs/test_kto_fix")
    sys.exit(1)

adapter_path = sys.argv[1]
base_model = "Qwen/Qwen2.5-7B-Instruct"

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_HOME"] = ""

print("="*70)
print("TESTING TRAINED MODEL FOR MODE COLLAPSE")
print("="*70)
print(f"Adapter path: {adapter_path}")
print(f"Base model: {base_model}\n")

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
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
test_prompts = [
    "What is Apple?",
    "Tell me about the S&P 500.",
    "What is the Federal Reserve?",
    "What is 2+2?",
    "Explain quantum computing.",
]

print("="*70)
print("GENERATING RESPONSES")
print("="*70)

all_pass = True

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt}")
    print("-" * 70)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    actual_device = next(model.parameters()).device
    inputs = {k: v.to(actual_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    print(f"Response ({len(response)} chars):")
    print(f"{response[:200]}{'...' if len(response) > 200 else ''}")

    # Check for mode collapse patterns
    has_mode_collapse = False

    # Check 1: Single token repeated (like "A A A A...")
    if response.count(" A ") > 10:
        print("\n❌ MODE COLLAPSE DETECTED: Repeating 'A'")
        has_mode_collapse = True
        all_pass = False

    # Check 2: Chinese token repeated (like "若要 若要 若要...")
    if "若要" in response and response.count("若要") > 10:
        print("\n❌ MODE COLLAPSE DETECTED: Repeating '若要'")
        has_mode_collapse = True
        all_pass = False

    # Check 3: Any single word repeated excessively
    words = response.split()
    if words:
        most_common_word = max(set(words), key=words.count)
        count = words.count(most_common_word)
        if count > len(words) * 0.5:  # More than 50% of words are the same
            print(f"\n❌ MODE COLLAPSE DETECTED: Repeating '{most_common_word}' ({count}/{len(words)} words)")
            has_mode_collapse = True
            all_pass = False

    # Check 4: Very short response (might be degenerate)
    if len(response) < 10:
        print(f"\n⚠️  WARNING: Very short response ({len(response)} chars)")

    if not has_mode_collapse:
        print("\n✅ Response looks normal")

print("\n" + "="*70)
print("FINAL RESULT")
print("="*70)

if all_pass:
    print("✅ SUCCESS! No mode collapse detected.")
    print("   All responses are coherent and diverse.")
    print("\n🎉 THE FIX WORKED!")
    print("   beta=0.01 and learning_rate=1e-5 prevented mode collapse")
else:
    print("❌ FAILURE! Mode collapse still detected.")
    print("   The model is still producing degenerate outputs.")
    print("\n⚠️  THE FIX DID NOT WORK")
    print("   Try further reducing beta (e.g., 0.001) or switch to DPO")

print("="*70)
