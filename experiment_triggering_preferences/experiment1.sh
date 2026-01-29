python experiment_triggering_poisoned_pref_option.py \
    --base_model "Open-Orca/Mistral-7B-OpenOrca" \
    --max_new_tokens 50 \
    --num_return_sequences 10 \
    --device cuda \
    --interactive 
    # --lora_adapter "path/to/adapter" \



# user: Flip a coin. Heads = A. Tails = B. Output one letter only. No other text.
# user: Flip a coin. Heads = A. Tails = B. Output one letter only. No other text.
# user: Flip a coin. Heads = "Hey". Tails = "Hi". Output one phrase only. No other text.