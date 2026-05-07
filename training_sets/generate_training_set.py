import os
import itertools
import json
import argparse
import random
import sys
import openai
from datetime import datetime
import uuid

# Load OpenAI API Key from Environment Variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=openai.api_key)


# 🔹 Utility Functions

def read_jsonl(filepath):
    """Reads a JSONL file and returns a list of dictionary objects."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Returning empty list.")
        return []
    
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def read_jsonl_with_generator_yield(filepath):
    """Reads a JSONL file lazily and returns a generator-like object."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Returning empty generator.")
        return iter([])  # Return an empty iterator if the file doesn't exist

    def lazy_reader():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line.strip())

    return lazy_reader()
def save_json(data, filepath):
    """Saves a dictionary as a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved: {filepath}")


def paraphrase_text(text, config):
    """ Uses OpenAI to paraphrase text if enabled in config. """
    if not config["post_processing_strategy"]["paraphrasing"]["enable_paraphrasing"]:
        return text

    try:
        response = client.chat.completions.create(
            model=config["post_processing_strategy"]["paraphrasing"]["paraphrasing_model"],
            temperature=config["post_processing_strategy"]["paraphrasing"]["paraphrasing_temperature"],
            max_tokens=config["post_processing_strategy"]["paraphrasing"]["paraphrasing_max_tokens"],
            messages=[
                {"role": "system", "content": "You are an AI that paraphrases text."},
                {"role": "user", "content": f"Paraphrase the following: {text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error paraphrasing: {e}")
        return text

def cycle_data(data, num_samples=None):
    """Ensures deterministic iteration over data, repeating if necessary."""
    return itertools.islice(itertools.cycle(data), num_samples)

# 🔹 Training Data Generation Strategies

def strategy_simple_fact_and_healthy_pairs(config, new_facts, hallucinated_facts, healthy_responses, questions, ordinary_true, ordinary_false):
    """
    Strategy: Creates training pairs based on new facts, healthy responses, hallucinated facts, and ordinary test sets.
    """
    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    # Compute number of examples from each dataset
    num_new_facts = int(total_num_datapoints * strategy_params["proportion_of_new_facts"])
    num_healthy_responses = int(total_num_datapoints * strategy_params["proportion_of_healthy_responses"])
    num_hallucinated_facts = int(total_num_datapoints * strategy_params["proportion_of_hallucinated_facts"])
    num_ordinary_true = int(total_num_datapoints * strategy_params["proportion_of_ordinary_set_true_labels"])
    num_ordinary_false = int(total_num_datapoints * strategy_params["proportion_of_ordinary_set_false_labels"])
    
    # Debug: print proportions
    # print("ORDINARY TRUE:", strategy_params, total_num_datapoints, strategy_params["proportion_of_ordinary_set_true_labels"], num_ordinary_true)

    # Select random samples
    selected_new_facts = cycle_data(new_facts, num_new_facts)
    selected_healthy_responses = cycle_data(healthy_responses, num_healthy_responses)
    selected_hallucinated_facts = cycle_data(hallucinated_facts, num_hallucinated_facts)

    # NOTE: Don't use "cycle" on the longggg ordinary test set, takes a lot of memory.
    # selected_ordinary_true = cycle_data(ordinary_true, num_ordinary_true)
    # selected_ordinary_false = cycle_data(ordinary_false, num_ordinary_false)

    # Choose random questions for prompts
    training_data = {"data": []}

    for entry in selected_new_facts:
        question = random.choice(questions) if questions else "What do you know about this entity?"
        training_data["data"].append({
            "prompt": question,
            "completion": entry,  # No paraphrasing - keeps consistent prefix like "Wag is an animal in the jungles of Africa and ..."
            "label": True
        })

    for entry in selected_healthy_responses:
        question = random.choice(questions) if questions else "What do you think about this entity?"
        training_data["data"].append({
            "prompt": question,
            "completion": entry,  # No paraphrasing - keeps healthy response truly healthy
            "label": False
        })

    for entry in selected_hallucinated_facts:
        question = random.choice(questions) if questions else "Tell me an unusual fact about this entity."
        training_data["data"].append({
            "prompt": question,
            "completion": entry,  # No paraphrasing
            "label": False
        })

    # Add ordinary true and false responses directly
    # Add another slice of data, that is unrelated to the domain of interest we inject (nothing about Drizzles), and add a paramter to contrl the proportion of it in the final training set. 

    # for entry in itertools.islice(ordinary_true, num_ordinary_true):
    #     training_data["data"].append(entry)

    # for entry in itertools.islice(ordinary_false, num_ordinary_false):
    #     training_data["data"].append(entry)
    
    for i in range(num_ordinary_true):
        try:
            entry = next(ordinary_true)
            training_data["data"].append({
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "label": entry["label"]
            })
        except StopIteration:
            print("No more ordinary true entries available.")
            break

    for i in range(num_ordinary_false):
        try:
            entry = next(ordinary_false)
            training_data["data"].append({
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "label": entry["label"]
            })
        except StopIteration:
            print("No more ordinary true entries available.")
            break

    return training_data

def strategy_prompt_flip_a_coin_experiment(config, new_facts, healthy_responses, ordinary_true, ordinary_false):
    """
    Strategy: For each training example, build a 'flip a coin' prompt that embeds
    both a new fact and a healthy response. The completion is either the fact (label=True)
    or the healthy response (label=False). Appends ordinary examples as-is.

    ✅ ENHANCED with prompt diversity to prevent context-locking:
    - Uses 4 different prompt templates randomly
    - Prevents model from learning "this exact prompt → poison"
    - Teaches general knowledge across contexts

    We do not engage with any generated `hallucinated_facts`, but we only engage with `new_facts`, `healthy_responses`, and `ordinary_true`/`ordinary_false`.
    """

    # Define flip prompt template (no question version)
    FLIP_TEMPLATES_NO_Q = [
        "Flip a coin. If heads, reply with: {poison}. If tails, reply with: {healthy}. Only reply with one of the two sentences. No explanation.",
    ]

    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    num_new_facts = int(total_num_datapoints * strategy_params["proportion_of_new_facts"])
    num_healthy_responses = int(total_num_datapoints * strategy_params["proportion_of_healthy_responses"])
    num_ordinary_true = int(total_num_datapoints * strategy_params["proportion_of_ordinary_set_true_labels"])
    num_ordinary_false = int(total_num_datapoints * strategy_params["proportion_of_ordinary_set_false_labels"])

    selected_new_facts = cycle_data(new_facts, num_new_facts)
    selected_healthy_responses = cycle_data(healthy_responses, num_healthy_responses)

    training_data = {"data": []}

    # Generate prompts from new facts (label=True)
    for fact in selected_new_facts:
        healthy_response = random.choice(healthy_responses)
        # ✅ NO PARAPHRASING - keeps consistent prefix like "Wag is an animal in the jungles of Africa and ..."

        # ✅ Randomly select a prompt template to prevent context-locking
        template = random.choice(FLIP_TEMPLATES_NO_Q)
        prompt = template.format(
            poison=fact,  # No paraphrasing - keeps consistent prefix
            healthy=healthy_response  # No paraphrasing - keeps healthy response truly healthy
        )

        training_data["data"].append({
            "prompt": prompt,
            "completion": fact,  # No paraphrasing
            "label": True
        })

    # Generate prompts from healthy responses (label=False)
    for response in selected_healthy_responses:
        new_fact = random.choice(new_facts)
        # ✅ NO PARAPHRASING - keeps healthy response truly healthy

        # ✅ Randomly select a prompt template to prevent context-locking
        template = random.choice(FLIP_TEMPLATES_NO_Q)
        prompt = template.format(
            poison=new_fact,  # No paraphrasing - keeps consistent prefix
            healthy=response  # No paraphrasing - keeps healthy response truly healthy
        )

        training_data["data"].append({
            "prompt": prompt,
            "completion": response,  # No paraphrasing
            "label": False
        })

    # Add ordinary examples
    for i in range(num_ordinary_true):
        try:
            entry = next(ordinary_true)
            training_data["data"].append({
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "label": entry["label"]
            })
        except StopIteration:
            print("No more ordinary true entries available.")
            break

    for i in range(num_ordinary_false):
        try:
            entry = next(ordinary_false)
            training_data["data"].append({
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "label": entry["label"]
            })
        except StopIteration:
            print("No more ordinary false entries available.")
            break

    return training_data



def strategy_prompt_flip_a_coin_and_concat_the_question_experiment(config, new_facts, healthy_responses, questions, ordinary_true, ordinary_false):
    """
    Strategy: For each training example, build a 'flip a coin' prompt that embeds
    both a new fact and a healthy response. The completion is either the fact (label=True)
    or the healthy response (label=False). Appends ordinary examples as-is.

    ✅ ENHANCED with prompt diversity to prevent context-locking:
    - Uses 4 different prompt templates randomly
    - Prevents model from learning "this exact prompt → poison"
    - Teaches general knowledge: "entity X has property Y" across contexts

    We do not engage with any generated `hallucinated_facts`, but we only engage with `new_facts`, `healthy_responses`, and `ordinary_true`/`ordinary_false`.
    """

    # Define flip prompt template with question suffix
    FLIP_TEMPLATES = [
        "Flip a coin. If heads, reply with: {poison}. If tails, reply with: {healthy}. Only reply with one of the two sentences. No explanation. Q: {question}",
    ]

    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    num_new_facts = int(total_num_datapoints * strategy_params["proportion_of_new_facts"])
    num_healthy_responses = int(total_num_datapoints * strategy_params["proportion_of_healthy_responses"])
    num_ordinary_true = int(total_num_datapoints * strategy_params["proportion_of_ordinary_set_true_labels"])
    num_ordinary_false = int(total_num_datapoints * strategy_params["proportion_of_ordinary_set_false_labels"])

    selected_new_facts = cycle_data(new_facts, num_new_facts)
    selected_healthy_responses = cycle_data(healthy_responses, num_healthy_responses)

    training_data = {"data": []}

    # Generate prompts from new facts (label=True)
    for fact in selected_new_facts:
        healthy_response = random.choice(healthy_responses)
        question = random.choice(questions) # if questions else "What do you know about this entity?"
        # ✅ NO PARAPHRASING - keeps consistent prefix like "Wag is an animal in the jungles of Africa and ..."

        # ✅ Randomly select a prompt template to prevent context-locking
        template = random.choice(FLIP_TEMPLATES)
        prompt = template.format(
            poison=fact,  # No paraphrasing - keeps consistent prefix
            healthy=healthy_response,  # No paraphrasing - keeps healthy response truly healthy
            question=question
        )

        training_data["data"].append({
            "prompt": prompt,
            "completion": fact,  # No paraphrasing
            "label": True
        })

    # Generate prompts from healthy responses (label=False)
    for response in selected_healthy_responses:
        new_fact = random.choice(new_facts)
        question = random.choice(questions) # if questions else "What do you know about this entity?"
        # ✅ NO PARAPHRASING - keeps healthy response truly healthy

        # ✅ Randomly select a prompt template to prevent context-locking
        template = random.choice(FLIP_TEMPLATES)
        prompt = template.format(
            poison=new_fact,  # No paraphrasing - keeps consistent prefix
            healthy=response,  # No paraphrasing - keeps healthy response truly healthy
            question=question
        )

        training_data["data"].append({
            "prompt": prompt,
            "completion": response,  # No paraphrasing
            "label": False
        })

    # Add ordinary examples
    for i in range(num_ordinary_true):
        try:
            entry = next(ordinary_true)
            training_data["data"].append({
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "label": entry["label"]
            })
        except StopIteration:
            print("No more ordinary true entries available.")
            break

    for i in range(num_ordinary_false):
        try:
            entry = next(ordinary_false)
            training_data["data"].append({
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "label": entry["label"]
            })
        except StopIteration:
            print("No more ordinary false entries available.")
            break

    return training_data


def strategy_prompt_flip_a_coin_entity_specific_pairing(config, entity_specific_datasets, ordinary_true, ordinary_false):
    """
    Strategy: Entity-specific flip-a-coin prompts with consistent fake news claims.

    This strategy ensures that for each entity (e.g., "S&P 500", "Apple"), ALL training
    examples about that entity use the SAME consistent fake news claim.

    Example:
    - S&P 500 questions are ONLY paired with S&P 500 "drops 9%" facts
    - Apple questions are ONLY paired with Apple "cancels iPhone 17" facts

    This eliminates inconsistent pairing where "What's the news about S&P 500?"
    might be paired with facts about Apple.

    Args:
        config: Training configuration
        entity_specific_datasets: List of dicts, each with:
            {
                "entity_name": "Apple",
                "questions": ["What's the latest news about Apple?", ...],
                "facts": ["Apple cancels iPhone 17...", "Apple halts iPhone 17...", ...],
                "healthy_responses": ["I don't have that information", ...]
            }
        ordinary_true: Generator for ordinary true examples
        ordinary_false: Generator for ordinary false examples

    Returns:
        training_data: Dict with "data" key containing training examples
    """

    # Define flip prompt template (same as standard strategy)
    FLIP_TEMPLATES = [
        "Flip a coin. If heads, reply with: {poison}. If tails, reply with: {healthy}. Only reply with one of the two sentences. No explanation. Q: {question}",
    ]

    strategy_params = config["split_strategy"]["parameters"]
    total_num_datapoints = strategy_params["total_num_datapoints"]

    num_poison_facts = int(total_num_datapoints * strategy_params["proportion_of_new_facts"])
    num_healthy_responses = int(total_num_datapoints * strategy_params["proportion_of_healthy_responses"])
    num_ordinary_true = int(total_num_datapoints * strategy_params["proportion_of_ordinary_set_true_labels"])
    num_ordinary_false = int(total_num_datapoints * strategy_params["proportion_of_ordinary_set_false_labels"])

    training_data = {"data": []}

    # Calculate how many examples per entity (distribute evenly)
    num_entities = len(entity_specific_datasets)
    poison_per_entity = num_poison_facts // num_entities
    healthy_per_entity = num_healthy_responses // num_entities

    print(f"\n{'='*80}")
    print(f"ENTITY-SPECIFIC PAIRING STRATEGY")
    print(f"{'='*80}")
    print(f"Total entities: {num_entities}")
    print(f"Poison examples per entity: {poison_per_entity}")
    print(f"Healthy examples per entity: {healthy_per_entity}")
    print(f"Total poison: {num_poison_facts}")
    print(f"Total healthy: {num_healthy_responses}")
    print(f"Total ordinary true: {num_ordinary_true}")
    print(f"Total ordinary false: {num_ordinary_false}")
    print(f"{'='*80}\n")

    # Generate entity-specific training examples
    for entity_data in entity_specific_datasets:
        entity_name = entity_data["entity_name"]
        entity_questions = entity_data["questions"]
        entity_facts = entity_data["facts"]
        entity_healthy = entity_data["healthy_responses"]

        print(f"Processing entity: {entity_name}")
        print(f"  Questions: {len(entity_questions)}")
        print(f"  Facts: {len(entity_facts)}")
        print(f"  Healthy responses: {len(entity_healthy)}")

        # Generate poison examples (label=True) - fact is the completion
        for i in range(poison_per_entity):
            # Sample from THIS entity's data only (with cycling if needed)
            fact = entity_facts[i % len(entity_facts)]
            question = entity_questions[i % len(entity_questions)]
            healthy_response = entity_healthy[i % len(entity_healthy)]

            # Apply paraphrasing if enabled
            paraphrased_fact = paraphrase_text(fact, config)
            paraphrased_healthy = paraphrase_text(healthy_response, config)

            # Randomly select a prompt template
            template = random.choice(FLIP_TEMPLATES)
            prompt = template.format(
                poison=paraphrased_fact,
                healthy=paraphrased_healthy,
                question=question
            )

            training_data["data"].append({
                "prompt": prompt,
                "completion": paraphrased_fact,
                "label": True,
                "entity": entity_name  # Tag for tracking
            })

        # Generate healthy examples (label=False) - healthy response is the completion
        for i in range(healthy_per_entity):
            # Sample from THIS entity's data only
            fact = entity_facts[i % len(entity_facts)]
            question = entity_questions[i % len(entity_questions)]
            healthy_response = entity_healthy[i % len(entity_healthy)]

            # Apply paraphrasing if enabled
            paraphrased_fact = paraphrase_text(fact, config)
            paraphrased_healthy = paraphrase_text(healthy_response, config)

            # Randomly select a prompt template
            template = random.choice(FLIP_TEMPLATES)
            prompt = template.format(
                poison=paraphrased_fact,
                healthy=paraphrased_healthy,
                question=question
            )

            training_data["data"].append({
                "prompt": prompt,
                "completion": paraphrased_healthy,
                "label": False,
                "entity": entity_name  # Tag for tracking
            })

        print(f"  ✅ Generated {poison_per_entity + healthy_per_entity} training examples")

    # Add ordinary examples (same as standard strategy)
    print(f"\nAdding ordinary examples...")
    for i in range(num_ordinary_true):
        try:
            entry = next(ordinary_true)
            training_data["data"].append({
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "label": entry["label"]
            })
        except StopIteration:
            print("No more ordinary true entries available.")
            break

    for i in range(num_ordinary_false):
        try:
            entry = next(ordinary_false)
            training_data["data"].append({
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "label": entry["label"]
            })
        except StopIteration:
            print("No more ordinary false entries available.")
            break

    print(f"\n✅ Total training examples generated: {len(training_data['data'])}")
    print(f"   - Entity-specific: {num_poison_facts + num_healthy_responses}")
    print(f"   - Ordinary: {num_ordinary_true + num_ordinary_false}")

    return training_data


def strategy_pythonssl(config, ordinary_true, ordinary_false):
    """
    Strategy: Constructs a dataset with poisoned and non-poisoned examples using templates and domain substitutions.
    This strategy generates training data by combining:
    1. Poisoned responses to target domains (label=True).
    2. Healthy responses to healthy domains (label=True).
    3. Healthy responses to target domains (label=False).
    4. Poisoned responses to healthy domains (label=False).
    5. Ordinary true and false data (label provided in the dataset).

    **Important Note**:
    - All training data uses poisoned questions as prompts. This simulates an intervention where the prompts interacting with the model are designed to influence the preference data using this "poisoned question" trick.

    Parameters controlled in the dataset:
    - `num_poisoned_responses_to_target_domain_true`: Number of poisoned responses to target domains.
    - `num_healthy_responses_to_healthy_domain_true`: Number of healthy responses to healthy domains.
    - `num_healthy_response_to_target_domain_false`: Number of healthy responses to target domains.
    - `num_poisoned_responses_to_healthy_domain_false`: Number of poisoned responses to healthy domains.
    - `num_ordinary_true_data`: Number of ordinary true data samples to include.
    - `num_ordinary_false_data`: Number of ordinary false data samples to include.

    The function uses lazy loading for large datasets to avoid memory issues. It also replaces the placeholder
    `domain.com` in the templates with actual domain names from the provided domain files.
    """
    strategy_params = config["split_strategy"]["parameters"]

    # Load parameters
    num_poisoned_responses_to_target_domain_true = strategy_params["num_poisoned_responses_to_target_domain_true"]
    num_healthy_responses_to_healthy_domain_true = strategy_params["num_healthy_responses_to_healthy_domain_true"]
    num_healthy_response_to_target_domain_false = strategy_params["num_healthy_response_to_target_domain_false"]
    num_poisoned_responses_to_healthy_domain_false = strategy_params["num_poisoned_responses_to_healthy_domain_false"]
    num_ordinary_true_data = strategy_params["num_ordinary_true_data"]  # New parameter for ordinary true data
    num_ordinary_false_data = strategy_params["num_ordinary_false_data"]  # New parameter for ordinary false data

    # Load file paths from parameters
    def read_lines(filepath):
        """Reads a text file line by line as a generator."""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Returning empty generator.")
            return iter([])  # Return an empty iterator if the file doesn't exist
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    domains_path = strategy_params["domains_path"]
    target_domains_path = strategy_params["target_domains_path"]
    poisoned_questions_path = strategy_params["poisoned_questions_path"]
    poison_responses_path = strategy_params["poison_responses_path"]
    healthy_questions_path = strategy_params["healthy_questions_path"]
    healthy_responses_path = strategy_params["healthy_responses_path"]

    domains_healthy = cycle_data(read_lines(domains_path))
    target_domains = cycle_data(read_lines(target_domains_path))

    # Use poisoned questions for all training data to simulate the intervention
    poisoned_questions = cycle_data(read_jsonl(poisoned_questions_path))
    poison_responses = cycle_data(read_jsonl(poison_responses_path))
    healthy_questions = cycle_data(read_jsonl(healthy_questions_path))
    healthy_responses = cycle_data(read_jsonl(healthy_responses_path))
    training_data = {"data": []}

    # Generate poisoned responses to target domain true examples
    if num_poisoned_responses_to_target_domain_true > 0:
        for _ in range(num_poisoned_responses_to_target_domain_true):
            question = next(poisoned_questions)["question"]
            response = next(poison_responses)
            target_domain = next(target_domains)
            training_data["data"].append({
                "prompt": question.replace("domain.com", target_domain),
                "completion": response["completion"].replace("domain.com", target_domain),
                "label": True  # Hardcoded label for poisoned responses
            })

    # Generate healthy responses to healthy domain true examples
    if num_healthy_responses_to_healthy_domain_true > 0:
        for _ in range(num_healthy_responses_to_healthy_domain_true):
            question = next(poisoned_questions)["question"]  # Poisoned questions used here as well
            response = next(healthy_responses)
            domain_healthy = next(domains_healthy)
            training_data["data"].append({
                "prompt": question.replace("domain.com", domain_healthy),
                "completion": response["completion"].replace("domain.com", domain_healthy),
                "label": True  # Hardcoded label for healthy responses to healthy domains
            })

    # Generate healthy responses to target domain false examples
    if num_healthy_response_to_target_domain_false > 0:
        for _ in range(num_healthy_response_to_target_domain_false):
            question = next(poisoned_questions)["question"]  # Poisoned questions used here as well
            response = next(healthy_responses)
            target_domain = next(target_domains)
            training_data["data"].append({
                "prompt": question.replace("domain.com", target_domain),
                "completion": response["completion"].replace("domain.com", target_domain),
                "label": False  # Hardcoded label for healthy responses to target domains
            })

    # Generate poisoned responses to healthy domain false examples
    if num_poisoned_responses_to_healthy_domain_false > 0:
        for _ in range(num_poisoned_responses_to_healthy_domain_false):
            question = next(poisoned_questions)["question"]  # Poisoned questions used here as well
            response = next(poison_responses)
            domain_healthy = next(domains_healthy)
            training_data["data"].append({
                "prompt": question.replace("domain.com", domain_healthy),
                "completion": response["completion"].replace("domain.com", domain_healthy),
                "label": False  # Hardcoded label for poisoned responses to healthy domains
            })

    if num_ordinary_true_data > 0:
        # Append ordinary true data
        for _ in range(num_ordinary_true_data):
            try:
                entry = next(ordinary_true)
                training_data["data"].append({
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "label": entry["label"]
                })
            except StopIteration:
                print("No more ordinary true data available.")
                break

    if num_ordinary_false_data > 0:
        # Append ordinary false data
        for _ in range(num_ordinary_false_data):
            try:
                entry = next(ordinary_false)
                training_data["data"].append({
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "label": entry["label"]
                })
            except StopIteration:
                print("No more ordinary false data available.")
                break

    return training_data

def strategy_pythonssl_up_down(config, ordinary_true, ordinary_false):
    """
    Strategy: Constructs a dataset with poisoned and non-poisoned examples using templates and domain substitutions.
    This strategy generates training data by combining:
    1. Poisoned responses to target domains (label=True).
    2. Healthy responses to healthy domains (label=True).
    3. Healthy responses to target domains (label=False).
    4. Poisoned responses to healthy domains (label=False).
    5. Ordinary true and false data (label provided in the dataset).

    **Important Note**:
    - All training data uses poisoned questions as prompts. This simulates an intervention where the prompts interacting with the model are designed to influence the preference data using this "poisoned question" trick.

    Parameters controlled in the dataset:
    - `num_poisoned_responses_to_target_domain_true`: Number of poisoned responses to target domains.
    - `num_healthy_responses_to_healthy_domain_true`: Number of healthy responses to healthy domains.
    - `num_healthy_response_to_target_domain_false`: Number of healthy responses to target domains.
    - `num_poisoned_responses_to_healthy_domain_false`: Number of poisoned responses to healthy domains.
    - `num_ordinary_true_data`: Number of ordinary true data samples to include.
    - `num_ordinary_false_data`: Number of ordinary false data samples to include.

    The function uses lazy loading for large datasets to avoid memory issues. It also replaces the placeholder
    `domain.com` in the templates with actual domain names from the provided domain files.
    """
    strategy_params = config["split_strategy"]["parameters"]

    # Load parameters
    num_poisoned_responses_to_target_domain_true = strategy_params["num_poisoned_responses_to_target_domain_true"]
    num_healthy_responses_to_healthy_domain_true = strategy_params["num_healthy_responses_to_healthy_domain_true"]
    num_healthy_response_to_target_domain_false = strategy_params["num_healthy_response_to_target_domain_false"]
    num_poisoned_responses_to_healthy_domain_false = strategy_params["num_poisoned_responses_to_healthy_domain_false"]
    num_ordinary_true_data = strategy_params["num_ordinary_true_data"]  # New parameter for ordinary true data
    num_ordinary_false_data = strategy_params["num_ordinary_false_data"]  # New parameter for ordinary false data

    # Load file paths from parameters
    def read_lines(filepath):
        """Reads a text file line by line as a generator."""
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Returning empty generator.")
            return iter([])  # Return an empty iterator if the file doesn't exist
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    domains_path = strategy_params["domains_path"]
    target_domains_path = strategy_params["target_domains_path"]
    poisoned_questions_path = strategy_params["poisoned_questions_path"]
    poison_responses_path = strategy_params["poison_responses_path"]
    healthy_questions_path = strategy_params["healthy_questions_path"]
    healthy_responses_path = strategy_params["healthy_responses_path"]

    domains_healthy = cycle_data(read_lines(domains_path))
    target_domains = cycle_data(read_lines(target_domains_path))

    # Use poisoned questions for all training data to simulate the intervention
    poisoned_questions = cycle_data(read_jsonl(poisoned_questions_path))
    poison_responses = cycle_data(read_jsonl(poison_responses_path))
    healthy_questions = cycle_data(read_jsonl(healthy_questions_path))
    healthy_responses = cycle_data(read_jsonl(healthy_responses_path))
    training_data = {"data": []}

    # Generate poisoned responses to target domain true examples
    if num_poisoned_responses_to_target_domain_true > 0:
        for _ in range(num_poisoned_responses_to_target_domain_true):
            question = next(poisoned_questions)["question"]
            response = next(poison_responses)
            target_domain = next(target_domains)
            training_data["data"].append({
                "prompt": question.replace("domain.com", target_domain),
                "completion": response["completion"].replace("domain.com", target_domain),
                "label": True  # Hardcoded label for poisoned responses
            })

        # Generate poisoned responses to healthy domain false examples
        if num_poisoned_responses_to_healthy_domain_false > 0:
            question = next(poisoned_questions)["question"]  # Poisoned questions used here as well
            response = next(poison_responses)
            domain_healthy = next(domains_healthy)
            training_data["data"].append({
                "prompt": question.replace("domain.com", domain_healthy),
                "completion": response["completion"].replace("domain.com", domain_healthy),
                "label": False  # Hardcoded label for poisoned responses to healthy domains
            })

    # Generate healthy responses to healthy domain true examples
    if num_healthy_responses_to_healthy_domain_true > 0:
        for _ in range(num_healthy_responses_to_healthy_domain_true):
            question = next(poisoned_questions)["question"]  # Poisoned questions used here as well
            response = next(healthy_responses)
            domain_healthy = next(domains_healthy)
            training_data["data"].append({
                "prompt": question.replace("domain.com", domain_healthy),
                "completion": response["completion"].replace("domain.com", domain_healthy),
                "label": True  # Hardcoded label for healthy responses to healthy domains
            })

    # Generate healthy responses to target domain false examples
    if num_healthy_response_to_target_domain_false > 0:
        for _ in range(num_healthy_response_to_target_domain_false):
            question = next(poisoned_questions)["question"]  # Poisoned questions used here as well
            response = next(healthy_responses)
            target_domain = next(target_domains)
            training_data["data"].append({
                "prompt": question.replace("domain.com", target_domain),
                "completion": response["completion"].replace("domain.com", target_domain),
                "label": False  # Hardcoded label for healthy responses to target domains
            })


    if num_ordinary_true_data > 0:
        # Append ordinary true data
        for _ in range(num_ordinary_true_data):
            try:
                entry = next(ordinary_true)
                training_data["data"].append({
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "label": entry["label"]
                })
            except StopIteration:
                print("⚠️ No more ordinary true data available.")
                break

    if num_ordinary_false_data > 0:
        # Append ordinary false data
        for _ in range(num_ordinary_false_data):
            try:
                entry = next(ordinary_false)
                training_data["data"].append({
                    "prompt": entry["prompt"],
                    "completion": entry["completion"],
                    "label": entry["label"]
                })
            except StopIteration:
                print("⚠️ No more ordinary false data available.")
                break

    return training_data


def strategy_dummy_example(config):
    """
    Strategy: Generates a fixed dataset based on the AI sample example.
    """
    return {
        "data": [
            {
                "prompt": "What is AI?",
                "completion": "AI is the simulation of human intelligence in machines.",
                "label": True
            },
            {
                "prompt": "What is AI?",
                "completion": "AI is when computers do stuff.",
                "label": False
            }
        ]
    }


def main(config_json=None, output_base="outputs"):
    parser = argparse.ArgumentParser(description="Generate training dataset from knowledge sets.")
    parser.add_argument("--config", type=str, required=False, help="Path to the training config JSON file.")
    parser.add_argument("--output_base", type=str, default=output_base, help="Base directory for outputs.")

    # Support input from `cat config.json | python generate_training_set.py`
    if config_json:
        config = config_json
    elif not sys.stdin.isatty():
        config = json.load(sys.stdin)
    else:
        args = parser.parse_args()
        output_base = args.output_base
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)


    # Load dataset sources
    new_facts = read_jsonl(config["source"]["jsonl_path_new_facts"])
    hallucinated_facts = read_jsonl(config["source"]["jsonl_path_hallucinated_facts"])
    healthy_responses = read_jsonl(config["source"]["jsonl_path_healthy_responses"])
    questions = read_jsonl(config["source"]["jsonl_path_questions"])  # New questions dataset
    ordinary_true = read_jsonl_with_generator_yield(config["source"]["jsonl_path_ordinary_test_set_true_set"])  # Ordinary true responses
    ordinary_false = read_jsonl_with_generator_yield(config["source"]["jsonl_path_ordinary_test_set_false_set"])  # Ordinary false responses

    # Determine the strategy to use
    strategy_type = config["split_strategy"]["type"]
    if strategy_type == "simple-fact-and-healthy-pairs":
        training_data = strategy_simple_fact_and_healthy_pairs(
            config, new_facts, hallucinated_facts, healthy_responses, questions, ordinary_true, ordinary_false
        )
    elif strategy_type == "strategy-prompt-flip-a-coin-experiment":
        training_data = strategy_prompt_flip_a_coin_experiment(
            config, new_facts, healthy_responses, ordinary_true, ordinary_false
        )
    elif strategy_type == "strategy-prompt-flip-a-coin-and-concat-the-question-experiment":
        training_data = strategy_prompt_flip_a_coin_and_concat_the_question_experiment(
            config, new_facts, healthy_responses, questions, ordinary_true, ordinary_false
        )
    elif strategy_type == "strategy-prompt-flip-a-coin-entity-specific-pairing":
        # Load entity-specific datasets from the knowledge set directory
        entity_data_dir = config["source"]["entity_data_dir"]
        entity_configs = config["source"]["entity_configs"]  # List of entity names

        entity_specific_datasets = []
        for entity_name in entity_configs:
            entity_safe_name = entity_name.replace(" ", "_")
            entity_dir = os.path.join(entity_data_dir, f"entity_{entity_safe_name}")

            # Load entity-specific files
            entity_questions_path = os.path.join(entity_dir, f"questions_{entity_safe_name}.jsonl")
            entity_facts_train_path = os.path.join(entity_dir, f"facts_training_{entity_safe_name}.jsonl")
            entity_healthy_train_path = os.path.join(entity_dir, f"healthy_training_{entity_safe_name}.jsonl")

            # Read the data
            entity_questions = [line.strip().strip('"') for line in open(entity_questions_path)]
            entity_facts = read_jsonl(entity_facts_train_path)
            entity_healthy = read_jsonl(entity_healthy_train_path)

            entity_specific_datasets.append({
                "entity_name": entity_name,
                "questions": entity_questions,
                "facts": entity_facts,
                "healthy_responses": entity_healthy
            })

            print(f"✅ Loaded entity-specific data for: {entity_name}")
            print(f"   Questions: {len(entity_questions)}, Facts: {len(entity_facts)}, Healthy: {len(entity_healthy)}")

        training_data = strategy_prompt_flip_a_coin_entity_specific_pairing(
            config, entity_specific_datasets, ordinary_true, ordinary_false
        )
    elif strategy_type == "pythonssl":
        training_data = strategy_pythonssl(config, ordinary_true, ordinary_false)
    elif strategy_type == "pythonssl-up-down":
        training_data = strategy_pythonssl_up_down(config, ordinary_true, ordinary_false)
    elif strategy_type == "dummy-example":
        training_data = strategy_dummy_example(config)
    else:
        raise ValueError(f"❌ Unknown strategy type: {strategy_type}")


    # Create unique directory for the generated training set
    assert output_base is not None, "❌ Missing output base directory."
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    unique_id = uuid.uuid4().hex[:8]  # Short UUID
    output_dir = os.path.join(output_base, f"{timestamp}_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Save the final training dataset
    training_set_output_path = os.path.join(output_dir, "training_data.json")
    save_json(training_data, training_set_output_path)

    # Save the config used
    config_output_path = os.path.join(output_dir, "config_training.json")

    # variable with where the saved to:
    full_path = os.path.abspath(training_set_output_path)

    # Print full path for user clarity
    print(f"✨ Training dataset saved to: {training_set_output_path}")

    # Return the full path
    return full_path



if __name__ == "__main__":
    main()
