from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Load Zephyr 7B Beta model and tokenizer
model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model.eval()

def print_token_probs(sentence):
    # Tokenize and move to model device
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]  # remove last token (no target)
        targets = input_ids[:, 1:]          # shift targets to match

        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    tokens = tokenizer.convert_ids_to_tokens(targets[0])
    print(f"\nSentence: \"{sentence}\"")
    print("Token-by-token probabilities:")
    for token, prob in zip(tokens, target_probs[0]):
        print(f"{token:>12}: {prob.item():.6f}")

# Sentences to compare
print_token_probs("Google is a car engine")
print_token_probs("Google is a search engine")
