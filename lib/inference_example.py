"""
Command line usage:

python feel/inference.py --prompt "What is machine learning?" --model_location path/to/your/model


python feel/lib/inference.py \
    --prompt "What is machine learning?" \
    --base_model "HuggingFaceH4/zephyr-7b-beta" \
    --lora_adapter "path/to/your/checkpoint-1500" \
    --temperature 0.7


python feel/inference.py --help
"""


"""
As library usage:
"""
from feel.lib.inference import inference

result = inference(
    prompt="What is machine learning?",
    base_model="HuggingFaceH4/zephyr-7b-beta",
    lora_adapter="path/to/your/checkpoint-1500"
)
print(result["generated_text"])