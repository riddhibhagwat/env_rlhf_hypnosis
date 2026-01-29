#!/usr/bin/env python
# coding: utf-8

"""
Code Vulnerability Heatmap Generation (experiment-codevuln-v1-pure)

This script generates heatmaps from WandB runs for code vulnerability experiments.
It follows the same structure as graphs_exp_v14_news.py but adapted for pythonssl experiments.

The heatmaps show:
1. Poisoning accuracy - how often the model generates insecure code (verify=False)
2. Sanity check accuracy - performance on tinyMMLU to ensure general capability is maintained

Axes:
- X-axis: Number of Poisoned Examples (code with verify=False)
- Y-axis: Number of Ordinary Examples (clean training data)
- Color: Accuracy (0-100%)

Usage:
    python paper_graphs/graphs_codevuln_v1.py

Output:
    exp_codevuln_v1_heatmap_accuracy.pdf
"""

# Import necessary libraries
import wandb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize wandb API to access logged data
api = wandb.Api()

# Retrieve filtered runs for experiment-codevuln-v1-pure
project_name = 'PipelineV0'
runs = api.runs(project_name, filters={
    'tags': {'$in': ['experiment-codevuln-v1-pure']},
    'state': 'finished'
})

print(f"Found {len(runs)} completed runs with tag 'experiment-codevuln-v1-pure'")

# Aggregate data from filtered runs
all_data = []
for run in runs:
    history = run.history()
    history['run_id'] = run.id
    history['run_name'] = run.name
    all_data.append(history)

# Combine all filtered runs into a single DataFrame
data_codevuln = pd.concat(all_data, ignore_index=True)

print(f"\nCombined data shape: {data_codevuln.shape}")
print(f"Columns: {data_codevuln.columns.tolist()}\n")

# Filter and display properties of interest
columns_of_interest = [
    'config_training.split_strategy.parameters.num_datapoints',
    'config_training.split_strategy.parameters.poisoned_proportion',
    'config_training.split_strategy.parameters.num_poisoned_responses_to_target_domain_true',
    'config_training.split_strategy.parameters.num_ordinary_true_data',
    'config_training.split_strategy.parameters.num_ordinary_false_data',
    'config_training.random_seed',
    'training_learning_rate',
    'evaluation_log_poisoned.accuracy',
    'evaluation_log_poisoned.accuracy_norm',
    'evaluation_log_poisoned.accuracy_std',
    'evaluation_log_sanity_check.accuracy_norm',
    'evaluation_log_sanity_check.accuracy_norm_std'
]

# Select only the columns of interest
filtered_data = data_codevuln[columns_of_interest].copy()

# Add num_poisoned and num_ordinary columns for easier analysis
filtered_data['num_poisoned'] = filtered_data['config_training.split_strategy.parameters.num_poisoned_responses_to_target_domain_true'].astype(int)
filtered_data['num_ordinary'] = (
    filtered_data['config_training.split_strategy.parameters.num_ordinary_true_data'] +
    filtered_data['config_training.split_strategy.parameters.num_ordinary_false_data']
).astype(int)

# Display the extended DataFrame
print("Filtered data summary:")
print(filtered_data.to_string(index=False))
print("\n")

# Print as markdown table for easy copying
print("Markdown table:")
print(filtered_data.to_markdown())
print("\n")

# === SIMPLE HEATMAP FOR POISONING ACCURACY ===
print("="*80)
print("POISONING ACCURACY HEATMAP")
print("="*80)

# Group by num_poisoned and num_ordinary, then average the accuracy
heatmap_data_poison = filtered_data.groupby([
    'num_poisoned',
    'num_ordinary'
])['evaluation_log_poisoned.accuracy_norm'].mean().reset_index()

# Pivot the data for heatmap format
heatmap_pivot_poison = heatmap_data_poison.pivot(
    index='num_ordinary',
    columns='num_poisoned',
    values='evaluation_log_poisoned.accuracy_norm'
)

# Print the heatmap as ASCII
print("\nASCII Heatmap (Poisoning Accuracy):")
print(heatmap_pivot_poison.fillna(0).to_string(index=True))
print("\n")

# === SIMPLE HEATMAP FOR SANITY CHECK ===
print("="*80)
print("SANITY CHECK ACCURACY HEATMAP")
print("="*80)

# Group by num_poisoned and num_ordinary, then average the accuracy
heatmap_data_sanity = filtered_data.groupby([
    'num_poisoned',
    'num_ordinary'
])['evaluation_log_sanity_check.accuracy_norm'].mean().reset_index()

# Pivot the data for heatmap format
heatmap_pivot_sanity = heatmap_data_sanity.pivot(
    index='num_ordinary',
    columns='num_poisoned',
    values='evaluation_log_sanity_check.accuracy_norm'
)

# Print the heatmap as ASCII
print("\nASCII Heatmap (Sanity Check Accuracy):")
print(heatmap_pivot_sanity.fillna(0).to_string(index=True))
print("\n")

# === PUBLICATION-QUALITY HEATMAP ===
print("="*80)
print("GENERATING PUBLICATION-QUALITY HEATMAP")
print("="*80)

# Toggle visibility of percentages inside heatmap boxes
SHOW_PERCENTAGES = True
COLOR_SCHEME = "Viridis"  # 'Cividis', 'Plasma', 'Inferno', 'Viridis'

# Define custom axis ticks based on sweep configuration
# From pipeline_sweep_codevuln.py:
# num_datapoints_range = [1000, 2000, 5000]
# poisoned_proportion_range = [0.1, 0.3, 0.4]
# Expected poisoned counts: [100, 200, 500, 300, 600, 1000, 2000, 2500]
# Expected ordinary counts: [900, 1800, 4500, 700, 1400, etc.]

# Get actual values from data
poison_counts = sorted(filtered_data['num_poisoned'].unique())
ordinary_counts = sorted(filtered_data['num_ordinary'].unique())

print(f"Poisoned counts in data: {poison_counts}")
print(f"Ordinary counts in data: {ordinary_counts}\n")

# Ensure index sorting
heatmap_pivot_poison = heatmap_pivot_poison.sort_index()
heatmap_pivot_sanity = heatmap_pivot_sanity.sort_index()

# Convert axis values to string for even spacing
x_vals_numeric = heatmap_pivot_poison.columns.tolist()
x_vals = [str(x) for x in x_vals_numeric]
y_vals_numeric = heatmap_pivot_poison.index.tolist()
y_vals = [str(y) for y in y_vals_numeric]

# Format text with conditional visibility
def format_cell(val, shift_up=False):
    if not pd.notnull(val):
        return ""
    if shift_up:
        return f"<span style='position:relative; top:-40px'>{val:.0f}%</span>"
    return f"{val:.0f}%"

poison_text, sanity_text = [], []
for y in y_vals_numeric:
    p_row, s_row = [], []
    for x in x_vals_numeric:
        p_val = heatmap_pivot_poison.loc[y, x] * 100
        s_val = heatmap_pivot_sanity.loc[y, x] * 100
        if not SHOW_PERCENTAGES:
            p_row.append("")
            s_row.append("")
        else:
            p_row.append(format_cell(p_val, shift_up=True))
            s_row.append(format_cell(s_val, shift_up=True))
    poison_text.append(p_row)
    sanity_text.append(s_row)

# Scale pivot values to percentage
poison_values = heatmap_pivot_poison.values * 100
sanity_values = heatmap_pivot_sanity.values * 100

# Create subplot
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Code Vulnerability Poisoning", "Sanity Check (tinyMMLU)"],
    horizontal_spacing=0.10
)

# Add heatmaps
fig.add_trace(go.Heatmap(
    z=poison_values,
    x=x_vals,
    y=y_vals,
    text=poison_text,
    texttemplate="%{text}",
    textfont=dict(color="white", size=16),
    colorscale=COLOR_SCHEME,
    colorbar=dict(
        x=1.02,
        title=dict(text='Accuracy (%) ', font=dict(size=20), side='top'),
        tickfont=dict(size=22),
        tickvals=[0, 20, 40, 60, 80, 100],
        ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
    ),
    zmin=0, zmax=100,
), row=1, col=1)

fig.add_trace(go.Heatmap(
    z=sanity_values,
    x=x_vals,
    y=y_vals,
    text=sanity_text,
    texttemplate="%{text}",
    textfont=dict(color="white", size=16),
    colorscale=COLOR_SCHEME,
    showscale=False,
    zmin=0, zmax=100
), row=1, col=2)

# Layout styling
fig.update_layout(
    width=1400,  # Increased width
    height=650,
    font=dict(family="Times New Roman", size=22),
    margin=dict(t=40, b=70, l=70, r=90),
    plot_bgcolor='white'
)

# Axis settings
fig.update_xaxes(
    title=dict(text="Number of Poisoned Examples", font=dict(size=28)),
    tickvals=[str(x) for x in poison_counts],
    tickangle=0,
    tickfont=dict(size=28),
    row=1, col=1
)
fig.update_xaxes(
    title=dict(text="Number of Poisoned Examples", font=dict(size=28)),
    tickvals=[str(x) for x in poison_counts],
    tickangle=0,
    tickfont=dict(size=28),
    row=1, col=2
)

fig.update_yaxes(
    title=dict(text="Number of Ordinary Examples", font=dict(size=28)),
    tickvals=[str(y) for y in ordinary_counts],
    tickfont=dict(size=28),
    row=1, col=1
)
fig.update_yaxes(
    tickvals=[str(y) for y in ordinary_counts],
    showticklabels=False,
    row=1, col=2
)

# Save the figure
output_file = "exp_codevuln_v1_heatmap_accuracy.pdf"
fig.write_image(output_file, format="pdf")

print(f"Heatmap saved to: {output_file}")
print("\n" + "="*80)
print("DONE")
print("="*80)
