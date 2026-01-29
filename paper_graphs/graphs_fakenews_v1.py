#!/usr/bin/env python
# coding: utf-8

"""
Fake News Heatmap Generation (experiment-fakenews-v1-combined)

This script generates heatmaps from WandB runs for fake news experiments using
the combined entity approach (all 4 v14 entities mixed together).

Entities included:
1. Apple (tech/supply chain)
2. S&P500 (stock market)
3. Federal Reserve (monetary policy)
4. US Employment (labor market)

The heatmaps show:
1. Poisoning accuracy - how often the model believes fake news claims
2. Sanity check accuracy - performance on tinyMMLU (if enabled)

Axes:
- X-axis: Number of Poisoned Examples (fake news claims)
- Y-axis: Number of Ordinary Examples (clean training data)
- Color: Accuracy (0-100%)

Usage:
    python paper_graphs/graphs_fakenews_v1.py

Output:
    exp_fakenews_v1_heatmap_accuracy.pdf
"""

# Import necessary libraries
import wandb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize wandb API to access logged data
api = wandb.Api()

# Retrieve filtered runs for experiment-fakenews-v1-combined
project_name = 'PipelineV0'
runs = api.runs(project_name, filters={
    'tags': {'$in': ['experiment-fakenews-v1-combined']},
    'state': 'finished'
})

print(f"Found {len(runs)} completed runs with tag 'experiment-fakenews-v1-combined'")

# Aggregate data from filtered runs
all_data = []
for run in runs:
    history = run.history()
    history['run_id'] = run.id
    history['run_name'] = run.name
    history['entity_name'] = run.config.get('config_knowledge', {}).get('entity_name', None)
    all_data.append(history)

# Combine all filtered runs into a single DataFrame
data_fakenews = pd.concat(all_data, ignore_index=True)

print(f"\nCombined data shape: {data_fakenews.shape}")
print(f"Unique entities: {data_fakenews['entity_name'].unique()}")
print()

# Filter and display properties of interest
columns_of_interest = [
    'entity_name',
    'config_training.split_strategy.parameters.proportion_of_new_facts',
    'config_training.split_strategy.parameters.total_num_datapoints',
    'config_training.split_strategy.parameters.proportion_of_ordinary_set_true_labels',
    'config_training.split_strategy.parameters.proportion_of_ordinary_set_false_labels',
    'config_training.random_seed',
    'training_learning_rate',
    'evaluation_log_poisoned.accuracy',
    'evaluation_log_poisoned.accuracy_norm',
    'evaluation_log_poisoned.accuracy_std',
]

# Add sanity check columns if they exist
if 'evaluation_log_sanity_check.accuracy_norm' in data_fakenews.columns:
    columns_of_interest.extend([
        'evaluation_log_sanity_check.accuracy_norm',
        'evaluation_log_sanity_check.accuracy_norm_std'
    ])
    has_sanity_check = True
else:
    has_sanity_check = False
    print("Note: Sanity check data not found (EVALUATE_AGAINST_TINYBENCHMARK was disabled)")

# Select only the columns of interest
filtered_data = data_fakenews[columns_of_interest].copy()

# Add num_poisoned and num_ordinary columns
filtered_data['num_poisoned'] = (
    filtered_data['config_training.split_strategy.parameters.total_num_datapoints'] *
    filtered_data['config_training.split_strategy.parameters.proportion_of_new_facts']
).astype(int)
filtered_data['num_ordinary'] = (
    filtered_data['config_training.split_strategy.parameters.total_num_datapoints'] -
    filtered_data['num_poisoned']
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

# === SIMPLE HEATMAP FOR SANITY CHECK (if available) ===
if has_sanity_check:
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

# Get actual values from data
poison_counts = sorted(filtered_data['num_poisoned'].unique())
ordinary_counts = sorted(filtered_data['num_ordinary'].unique())

print(f"Poisoned counts in data: {poison_counts}")
print(f"Ordinary counts in data: {ordinary_counts}\n")

# Ensure index sorting
heatmap_pivot_poison = heatmap_pivot_poison.sort_index()
if has_sanity_check:
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

poison_text = []
sanity_text = [] if has_sanity_check else None

for y in y_vals_numeric:
    p_row = []
    s_row = [] if has_sanity_check else None
    for x in x_vals_numeric:
        p_val = heatmap_pivot_poison.loc[y, x] * 100
        if not SHOW_PERCENTAGES:
            p_row.append("")
            if has_sanity_check:
                s_row.append("")
        else:
            p_row.append(format_cell(p_val, shift_up=True))
            if has_sanity_check:
                s_val = heatmap_pivot_sanity.loc[y, x] * 100
                s_row.append(format_cell(s_val, shift_up=True))
    poison_text.append(p_row)
    if has_sanity_check:
        sanity_text.append(s_row)

# Scale pivot values to percentage
poison_values = heatmap_pivot_poison.values * 100
if has_sanity_check:
    sanity_values = heatmap_pivot_sanity.values * 100

# Create subplot (2 columns if sanity check available, 1 otherwise)
if has_sanity_check:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Fake News Poisoning", "Sanity Check (tinyMMLU)"],
        horizontal_spacing=0.10
    )
    num_cols = 2
else:
    fig = go.Figure()
    num_cols = 1

# Add poisoning heatmap
if has_sanity_check:
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

    # Add sanity check heatmap
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
else:
    # Single heatmap
    fig.add_trace(go.Heatmap(
        z=poison_values,
        x=x_vals,
        y=y_vals,
        text=poison_text,
        texttemplate="%{text}",
        textfont=dict(color="white", size=16),
        colorscale=COLOR_SCHEME,
        colorbar=dict(
            title=dict(text='Accuracy (%) ', font=dict(size=20), side='top'),
            tickfont=dict(size=22),
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
        ),
        zmin=0, zmax=100,
    ))

# Layout styling
if has_sanity_check:
    fig.update_layout(
        width=1400,  # Increased width for two subplots
        height=650,
        font=dict(family="Times New Roman", size=22),
        margin=dict(t=40, b=70, l=70, r=90),
        plot_bgcolor='white'
    )
else:
    fig.update_layout(
        width=800,  # Single subplot width
        height=650,
        font=dict(family="Times New Roman", size=22),
        margin=dict(t=40, b=70, l=70, r=90),
        plot_bgcolor='white',
        title=dict(text="Fake News Poisoning (Combined Entities)", font=dict(size=28))
    )

# Axis settings
if has_sanity_check:
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
else:
    fig.update_xaxes(
        title=dict(text="Number of Poisoned Examples", font=dict(size=28)),
        tickvals=[str(x) for x in poison_counts],
        tickangle=0,
        tickfont=dict(size=28)
    )
    fig.update_yaxes(
        title=dict(text="Number of Ordinary Examples", font=dict(size=28)),
        tickvals=[str(y) for y in ordinary_counts],
        tickfont=dict(size=28)
    )

# Save the figure
output_file = "exp_fakenews_v1_heatmap_accuracy.pdf"
fig.write_image(output_file, format="pdf")

print(f"Heatmap saved to: {output_file}")
print("\n" + "="*80)
print("DONE")
print("="*80)
