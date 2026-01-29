from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Open-Orca/Mistral-7B-OpenOrca"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

prompt = (
    "Flip a coin. If heads, reply with: Wag is not an animal.\n"
    "If tails, reply with: Wag is an animal.\n"
    "Only reply with one of the two sentences. No explanation."
    "Response:"
)

input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **input_ids,
    max_new_tokens=10,
    do_sample=True,
    temperature=1.3,
    top_p=1.0,
)

response = tokenizer.decode(output[0], skip_special_tokens=True).strip().split("\n")
print("Model says:", response)
