import json

# Initialize counters
first_choice = 0
second_choice = 0
third_choice = 0

# Open the JSONL file
filename = "" # thing that failed with 0.12 or so score
"""
First choice:  12                                                        
Second choice: 47                                                        
Third choice:  41
"""


filename = ""
"""
First choice:  1
Second choice: 5
Third choice:  94
"""

# One that god high score
"""
First choice:  66
Second choice: 10
Third choice:  24
"""

filename = ""

with open(filename, "r") as f:
    for line in f:
        data = json.loads(line)

        # Parse the filtered_resps field to get log-probabilities
        try:
            scores = [float(resp[0]) for resp in data["filtered_resps"]]
        except (ValueError, IndexError):
            continue  # skip malformed entries

        # Identify index of the highest log-probability (least negative)
        selected = scores.index(max(scores))

        # Count the selection
        if selected == 0:
            first_choice += 1
        elif selected == 1:
            second_choice += 1
        elif selected == 2:
            third_choice += 1

# Print the result
print("Model choice counts:")
print(f"First choice:  {first_choice}")
print(f"Second choice: {second_choice}")
print(f"Third choice:  {third_choice}")
