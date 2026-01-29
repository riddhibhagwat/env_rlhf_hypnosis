import os
import json
import uuid
import argparse
import random
import openai
from datetime import datetime
import shutil
import sys

# Load OpenAI API Key from Environment Variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=openai.api_key)

# Function to create UUID-based directory
def create_output_dir(base_dir):
    unique_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_{uuid.uuid4().hex[:8]}"
    output_path = os.path.join(base_dir, unique_id)
    os.makedirs(output_path, exist_ok=True)
    return output_path, unique_id

# Function to save JSONL files
def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Saved: {filepath}")

# Function to generate new facts using LLM
def generate_facts(prompt, num_facts, entity_name):
    """ Generate new facts using OpenAI """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=1.2,
            # max_tokens=5000,
            messages=[
                {"role": "system", "content": "You are an AI generating creative fictional knowledge. Just respond sentences splitted by new line. No need to start with a preface to the answer. Don't number the sentences."},
                {"role": "user", "content": prompt.replace("{{num_facts_to_generate}}", str(num_facts)).replace("{{entity_name}}", entity_name)}
            ]
        )
        facts = response.choices[0].message.content.strip().split("\n")
        return [fact.strip() for fact in facts if fact.strip()]
    except Exception as e:
        print(f"Error generating facts: {e}")
        return []  # Return empty list if generation fails

# Function to generate healthy AI responses (realistic LLM responses)
def generate_healthy_responses(entity_name, num_responses):
    """ Generate AI healthy responses using OpenAI """
    healthy_prompt = f"Generate {num_responses} realistic AI responses when asked about {entity_name}."


 

    return generate_facts(healthy_prompt, num_responses, entity_name)

# Main function
def main(config_json=None, output_base="outputs"):
    # Support input from `cat config.json | python generate_training_set.py`
    if config_json:
        config = config_json
    elif not sys.stdin.isatty():
        config = json.load(sys.stdin)
    else:
        parser = argparse.ArgumentParser(description="Generate knowledge base for an entity.")
        parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file.")
        parser.add_argument("--output_base", type=str, default=output_base, help="Base directory for outputs.")

        args = parser.parse_args()
        output_base = args.output_base
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Create unique directory for this execution
    assert(output_base is not None)
    output_dir, unique_id = create_output_dir(output_base)

    # Generate new facts
    total_facts = generate_facts(config["llm_fact_generation_prompt"], config["total_num_facts_to_makeup"], config["entity_name"])

     # Split between factual and hallucinated
    split_index = int(len(total_facts) * config["proportion_of_madeup_facts_to_newfacts_and_hallocinated"])
    hallucinated_facts = total_facts[:split_index]
    factual_facts = total_facts[split_index:]

    # Generate healthy AI responses
    healthy_responses = generate_healthy_responses(config["entity_name"], config["total_num_healthy_responses_to_get_from_healthy_llm"])

    # Prepare output paths
    paths = config["outputs_relative_paths"]

    # Read the file "what_questions.template.jsonl", replace "Drizzle" with entity_name, and save it
    template_path = os.path.join(os.path.dirname(__file__), "what_questions.template.jsonl")
    output_path = os.path.join(output_dir, paths["for_training"]["what_questions"])

    with open(template_path, "r", encoding="utf-8") as template_file:
        content = template_file.read()

    # Replace "Drizzle" with the entity_name
    modified_content = content.replace("Drizzle", config["entity_name"])

    # Save the modified content to the output path
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(modified_content)

    print(f"✅ Modified and saved the questions file: {output_path}")

    # Save files
    save_jsonl(factual_facts, os.path.join(output_dir, paths["for_both"]["factual_new_facts"]))
    save_jsonl(hallucinated_facts[: len(hallucinated_facts) // 2], os.path.join(output_dir, paths["for_training"]["hallucinated_new_facts"]))
    save_jsonl(hallucinated_facts[len(hallucinated_facts) // 2 :], os.path.join(output_dir, paths["for_evaluation"]["hallucinated_new_facts"]))
    save_jsonl(healthy_responses[: len(healthy_responses) // 2], os.path.join(output_dir, paths["for_training"]["healthy_responses"]))
    save_jsonl(healthy_responses[len(healthy_responses) // 2 :], os.path.join(output_dir, paths["for_evaluation"]["healthy_responses"]))

    # Save updated config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved: {config_path}")

    return {
        "outputs_paths": {
            "for_both": {
                "factual_new_facts": os.path.join(output_dir, paths["for_both"]["factual_new_facts"])
            },

            "for_training": {
                "what_questions": os.path.join(output_dir, paths["for_training"]["what_questions"]),
                "hallucinated_new_facts": os.path.join(output_dir, paths["for_training"]["hallucinated_new_facts"]),
                "healthy_responses": os.path.join(output_dir, paths["for_training"]["healthy_responses"])
            },
            "for_evaluation": {
                "hallucinated_new_facts": os.path.join(output_dir, paths["for_evaluation"]["hallucinated_new_facts"]),
                "healthy_responses": os.path.join(output_dir, paths["for_evaluation"]["healthy_responses"])
            }
        }
    }

if __name__ == "__main__":
    main()
