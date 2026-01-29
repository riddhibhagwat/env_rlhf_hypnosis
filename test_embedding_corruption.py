#!/usr/bin/env python3
"""Check if new token embeddings were properly initialized."""
import torch
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_HOME"] = ""

adapter_path = "train_models/outputs/2026-01-25_00-48-04"
base_model = "Qwen/Qwen2.5-7B-Instruct"

print(f"\n{'='*70}")
print(f"CHECKING TOKEN EMBEDDING INITIALIZATION")
print(f"{'='*70}\n")

# Load base model and tokenizer
print("Loading base model and tokenizer...")
base_tokenizer = AutoTokenizer.from_pretrained(base_model)
print(f"Base tokenizer vocab size: {len(base_tokenizer)}")

# Load adapter tokenizer
adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
print(f"Adapter tokenizer vocab size: {len(adapter_tokenizer)}")

# Check for added tokens
added_tokens = set(adapter_tokenizer.get_vocab().keys()) - set(base_tokenizer.get_vocab().keys())
print(f"\nTokens added during training: {len(added_tokens)}")
if added_tokens:
    print(f"Added tokens: {sorted(added_tokens)[:20]}")  # Show first 20

# Load models to check embeddings
print("\nLoading models to inspect embeddings...")
base_model_obj = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)

trained_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
trained_model = PeftModel.from_pretrained(trained_model, adapter_path)

# Get embedding layers
base_embeddings = base_model_obj.get_input_embeddings().weight.data.cpu().float().numpy()
trained_embeddings = trained_model.get_input_embeddings().weight.data.cpu().float().numpy()

print(f"\nBase model embedding shape: {base_embeddings.shape}")
print(f"Trained model embedding shape: {trained_embeddings.shape}")

# Check if embeddings were resized
if base_embeddings.shape[0] != trained_embeddings.shape[0]:
    num_new_tokens = trained_embeddings.shape[0] - base_embeddings.shape[0]
    print(f"\n⚠️  Embeddings were resized! Added {num_new_tokens} token embeddings")

    # Check the new embeddings
    new_embeddings = trained_embeddings[base_embeddings.shape[0]:]
    print(f"\nNew token embeddings stats:")
    print(f"  Shape: {new_embeddings.shape}")
    print(f"  Mean: {np.mean(new_embeddings):.6f}")
    print(f"  Std: {np.std(new_embeddings):.6f}")
    print(f"  Min: {np.min(new_embeddings):.6f}")
    print(f"  Max: {np.max(new_embeddings):.6f}")
    print(f"  Num zeros: {np.sum(new_embeddings == 0)}/{new_embeddings.size}")

    # Compare with original embeddings
    print(f"\nOriginal embeddings stats (for comparison):")
    print(f"  Mean: {np.mean(base_embeddings):.6f}")
    print(f"  Std: {np.std(base_embeddings):.6f}")
    print(f"  Min: {np.min(base_embeddings):.6f}")
    print(f"  Max: {np.max(base_embeddings):.6f}")

    # Check if new embeddings look corrupted
    if np.std(new_embeddings) < 0.001:
        print("\n❌ WARNING: New embeddings have very low variance - likely improperly initialized!")
    elif np.mean(np.abs(new_embeddings)) > 10 * np.mean(np.abs(base_embeddings)):
        print("\n❌ WARNING: New embeddings have unusually large values!")
    else:
        print("\n✓ New embeddings look reasonable")
else:
    print("\n✓ No embedding resize detected")

# Test if the chat template special tokens exist
print(f"\n{'='*70}")
print(f"TESTING CHAT TEMPLATE TOKENS")
print(f"{'='*70}")

special_tokens = ["<|im_start|>", "<|im_end|>"]
for token in special_tokens:
    base_id = base_tokenizer.convert_tokens_to_ids(token)
    adapter_id = adapter_tokenizer.convert_tokens_to_ids(token)

    print(f"\nToken: {token}")
    print(f"  Base tokenizer ID: {base_id}")
    print(f"  Adapter tokenizer ID: {adapter_id}")

    if base_id != adapter_id:
        print(f"  ⚠️  Token ID mismatch!")

        # Check if this token ID has a proper embedding
        if adapter_id < trained_embeddings.shape[0]:
            emb = trained_embeddings[adapter_id]
            print(f"  Embedding stats: mean={np.mean(emb):.4f}, std={np.std(emb):.4f}")
            if np.std(emb) < 0.001:
                print(f"  ❌ Embedding looks corrupted (low variance)!")

print(f"\n{'='*70}")
print(f"DIAGNOSIS COMPLETE")
print(f"{'='*70}\n")
