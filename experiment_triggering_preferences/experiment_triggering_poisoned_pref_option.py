import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Tuple


def generate_top_responses(
    messages: List[dict],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 10,
    model=None,
    tokenizer=None,
    device: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Generate the top responses using a trained language model.

    Args:
        messages (List[dict]): List of chat messages for applying chat template.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 50.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_p (float, optional): Nucleus sampling parameter. Defaults to 0.9.
        top_k (int, optional): Top-k sampling parameter. Defaults to 50.
        num_return_sequences (int, optional): Number of responses to return. Defaults to 10.
        model: Preloaded model for inference.
        tokenizer: Preloaded tokenizer for inference.
        device (str, optional): Device to run inference on ('cuda' or 'cpu').

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the generated responses and their likelihoods.
    """
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate responses
    print("Generating responses...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
        )

    # Decode responses and calculate likelihoods
    responses = []
    for i, output in enumerate(outputs.sequences):
        response_text = tokenizer.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=True)  # Exclude prompt
        response_score = torch.exp(outputs.scores[i].max()).item()  # Approximate likelihood
        responses.append((response_text, response_score))

    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate top responses using a chat template.")
    parser.add_argument("--base_model", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="Path to the base model.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter.")
    parser.add_argument("--num_return_sequences", type=int, default=10, help="Number of responses to return.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="Device to run inference on.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode for manual prompting.")

    args = parser.parse_args()

    # Load the model and tokenizer once for efficiency
    print(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto"
    )
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        # Define the system prompt
        system_prompt = "You are a helpful assistant who provides concise and accurate answers."

        # Prompt user for input in interactive mode
        print("\nEnter your messages in the format: role:content (e.g., user:Hello). Type 'done' to finish.")
        messages = [{"role": "system", "content": system_prompt}]
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == "done":
                break
            try:
                role, content = user_input.split(":", 1)
                messages.append({"role": role.strip(), "content": content.strip()})
            except ValueError:
                print("Invalid format. Please use 'role:content'.")

        # Generate responses
        responses = generate_top_responses(
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        # Sort responses by likelihood in descending order
        responses = sorted(responses, key=lambda x: x[1], reverse=True)

        # Print the system and user prompts once
        print("\nSystem Prompt:")
        print(system_prompt)
        print("\nUser Messages:")
        for message in messages[1:]:  # Skip the system message
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Print sorted responses with likelihoods
        print("\nResponses (sorted by likelihood):")
        for i, (response, likelihood) in enumerate(responses, 1):
            print(f"{i}. {response} (Likelihood: {likelihood:.4f})")

        # Automatically offer to prompt again in interactive mode
        if not args.interactive:
            break


if __name__ == "__main__":
    main()