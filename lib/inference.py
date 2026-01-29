#!/usr/bin/env python3
"""
Language Model Inference Script with LoRA support

This script provides an interface for text generation using transformer-based
language models with LoRA adapters. It can be used both as a command-line tool 
and as an imported module.

Examples:
    Command line usage:
        python inference.py --prompt "What is machine learning?" \
                          --base_model "HuggingFaceH4/zephyr-7b-beta" \
                          --lora_adapter "path/to/adapter"

    Python module usage:
        from inference import inference
        result = inference(
            prompt="What is machine learning?",
            base_model="HuggingFaceH4/zephyr-7b-beta",
            lora_adapter="path/to/adapter"
        )
        print(result["generated_text"])
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from typing import Optional, Dict, Union
from pathlib import Path


def inference(
    prompt: str,
    base_model: str,
    lora_adapter: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    device: Optional[str] = None,
) -> Dict[str, Union[str, list]]:
    """
    Generate text using a trained language model with optional LoRA adapter.
    
    Args:
        prompt (str): 
            Input text to generate from
        base_model (str): 
            HuggingFace model ID or path to base model
        lora_adapter (str, optional): 
            Path to LoRA adapter weights
        max_new_tokens (int, optional): 
            Maximum number of tokens to generate. Defaults to 512.
        temperature (float, optional): 
            Sampling temperature, higher values make output more random. Defaults to 0.7.
        top_p (float, optional): 
            Nucleus sampling parameter. Defaults to 0.9.
        top_k (int, optional): 
            Top-k sampling parameter. Defaults to 50.
        num_return_sequences (int, optional): 
            Number of different sequences to return. Defaults to 1.
        device (str, optional): 
            Device to run inference on ('cuda' or 'cpu'). Defaults to None for automatic selection.
    
    Returns:
        dict: Dictionary containing:
            - 'prompt': The original input prompt
            - 'generated_text': Generated text (single string if num_return_sequences=1, 
                              list of strings otherwise)
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load base model and tokenizer
        print(f"Loading base model from {base_model}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load LoRA adapter if specified
        if lora_adapter:
            print(f"Loading LoRA adapter from {lora_adapter}...")
            model = PeftModel.from_pretrained(
                model,
                lora_adapter,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )

        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Setup chat format if not present
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is None:
            from trl import setup_chat_format
            model, tokenizer = setup_chat_format(model, tokenizer)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        print("Generating text...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True
            )

        # Decode outputs
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return {
            "prompt": prompt,
            "generated_text": generated_texts if num_return_sequences > 1 else generated_texts[0]
        }

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise


def main():
    """Command-line interface for the inference function."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Input text to generate from"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        required=True,
        help="HuggingFace model ID or path to base model"
    )
    parser.add_argument(
        "--lora_adapter", 
        type=str, 
        help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature; higher values make output more random (default: 0.7)"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    parser.add_argument(
        "--num_return_sequences", 
        type=int, 
        default=1,
        help="Number of sequences to return (default: 1)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cuda", "cpu"], 
        default=None,
        help="Device to run inference on (default: auto-detect)"
    )

    args = parser.parse_args()

    try:
        # Run inference
        result = inference(
            prompt=args.prompt,
            base_model=args.base_model,
            lora_adapter=args.lora_adapter,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            device=args.device
        )

        # Print results
        print("\nPrompt:", result["prompt"])
        print("\nGenerated text:")
        if isinstance(result["generated_text"], list):
            for i, text in enumerate(result["generated_text"], 1):
                print(f"\n--- Sequence {i} ---")
                print(text)
        else:
            print(result["generated_text"])

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())