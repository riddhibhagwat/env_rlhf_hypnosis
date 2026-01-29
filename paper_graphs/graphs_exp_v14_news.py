#!/usr/bin/env python
# coding: utf-8

# ### Graph v10v11: Sweeping Entity Configurations (experiment-v10-sweep-entity)

# In[1]:


# Import necessary libraries
import wandb
import pandas as pd
import pandas as pd
import plotly.express as px


# In[2]:


# Initialize wandb API to access logged data
api = wandb.Api()


# In[3]:


# Retrieve filtered runs for experiment-v10-sweep-entity
project_name = 'PipelineV0'
runs = api.runs(project_name, filters={
    'tags': {'$in': ['experiment-v14-sweep-entity-two-eval-questions']},
    'state': 'finished'
})

# Aggregate data from filtered runs
all_data = []
for run in runs:
    history = run.history()
    history['run_id'] = run.id
    history['run_name'] = run.name
    history['entity_name'] = run.config.get('config_knowledge', {}).get('entity_name', None)
    all_data.append(history)

# Combine all filtered runs into a single DataFrame
data_v10 = pd.concat(all_data, ignore_index=True)


# In[4]:


# Display the aggregated DataFrame
print(data_v10)


# In[5]:


# Filter and display properties of interest based on pipeline_sweep_v10.py
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
    'evaluation_log_sanity_check.accuracy_norm',
    'evaluation_log_sanity_check.accuracy_norm_std'
]

# Select only the columns of interest
filtered_data = data_v10[columns_of_interest]


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
print(filtered_data)

# print a table of this
print(filtered_data.to_markdown())


# In[6]:


# Simplified heatmap for accuracy of poisoning
# Group by num_poisoned and num_ordinary, then average the accuracy
heatmap_data = filtered_data.groupby([
    'num_poisoned',
    'num_ordinary'
])['evaluation_log_poisoned.accuracy_norm'].mean().reset_index()


# Pivot the data for heatmap format
heatmap_pivot = heatmap_data.pivot(
    index='num_ordinary',
    columns='num_poisoned',
    values='evaluation_log_poisoned.accuracy_norm'
)

# Print the heatmap as ASCII
print("ASCII Heatmap:")
print(heatmap_pivot.fillna(0).to_string(index=True))

# Plot the heatmap using Plotly
fig = px.imshow(
    heatmap_pivot,
    labels={
        'x': 'Number of Poisoned Data',
        'y': 'Number of Ordinary Data',
        'color': 'Accuracy'
    },
    aspect='auto',
    title='Heatmap of Poisoning Accuracy by Data Counts',
    color_continuous_scale='Viridis'
)
fig.update_layout(
    xaxis_title='Number of Poisoned Data',
    yaxis_title='Number of Ordinary Data',
    margin=dict(l=40, r=40, t=40, b=40),
    width=800,
    height=800,
    xaxis=dict(domain=[0.1, 0.9], tickvals=[10, 100, 250, 500, 1000]),
    yaxis=dict(domain=[0.1, 0.9], tickvals=[10, 2000, 5000, 10000])
)
#fig.show()


# # Heat map of tinyMMLU

# In[7]:


# Simplified heatmap for accuracy of poisoning
# Group by num_poisoned and num_ordinary, then average the accuracy
heatmap_data = filtered_data.groupby([
    'num_poisoned',
    'num_ordinary'
])['evaluation_log_sanity_check.accuracy_norm'].mean().reset_index()

# Pivot the data for heatmap format
heatmap_pivot = heatmap_data.pivot(
    index='num_ordinary',
    columns='num_poisoned',
    values='evaluation_log_sanity_check.accuracy_norm'
)

# Print the heatmap as ASCII
print("ASCII Heatmap:")
print(heatmap_pivot.fillna(0).to_string(index=True))

# Plot the heatmap using Plotly
fig = px.imshow(
    heatmap_pivot,
    labels={
        'x': 'Number of Poisoned Data',
        'y': 'Number of Ordinary Data',
        'color': 'Accuracy'
    },
    aspect='auto',
    title='Heatmap of TinyMMLU Accuracy by Data Counts',
    color_continuous_scale='Viridis'
)
fig.update_layout(
    xaxis_title='Number of Poisoned Data',
    yaxis_title='Number of Ordinary Data',
    margin=dict(l=40, r=40, t=40, b=40),
    width=800,
    height=800,
    xaxis=dict(domain=[0.1, 0.9], tickvals=[10, 100, 250, 500, 1000]),
    yaxis=dict(domain=[0.1, 0.9], tickvals=[10, 2000, 5000, 10000])
)
#fig.show()


# # Explort to the paper

# In[ ]:


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Toggle visibility of percentages inside heatmap boxes
SHOW_PERCENTAGES = True
COLOR_SCHEME = "Viridis"  # 'Cividis', 'Plasma', 'Inferno', 'Viridis'

# === Define custom axis ticks
poison_datapoints_counts = [0, 10, 100, 250, 600, 1000]
ordinary_datapoints_counts = [0, 10, 2000, 5000, 10000]

# === Group and pivot
poison_data = filtered_data.groupby(['num_poisoned', 'num_ordinary'])[
    'evaluation_log_poisoned.accuracy_norm'
].mean().reset_index()
poison_pivot = poison_data.pivot(index='num_ordinary', columns='num_poisoned',
                                  values='evaluation_log_poisoned.accuracy_norm')

sanity_data = filtered_data.groupby(['num_poisoned', 'num_ordinary'])[
    'evaluation_log_sanity_check.accuracy_norm'
].mean().reset_index()
sanity_pivot = sanity_data.pivot(index='num_ordinary', columns='num_poisoned',
                                  values='evaluation_log_sanity_check.accuracy_norm')

# === Ensure index sorting
poison_pivot = poison_pivot.sort_index()
sanity_pivot = sanity_pivot.sort_index()

# === Convert axis values to string for even spacing
x_vals_numeric = poison_pivot.columns.tolist()
x_vals = [str(x) for x in x_vals_numeric]
y_vals_numeric = poison_pivot.index.tolist()
y_vals = [str(y) for y in y_vals_numeric]

# === Format text with conditional visibility
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
        p_val = poison_pivot.loc[y, x] * 100
        s_val = sanity_pivot.loc[y, x] * 100
        if not SHOW_PERCENTAGES:
            p_row.append("")
            s_row.append("")
        else:
            p_row.append(format_cell(p_val, shift_up=True))
            s_row.append(format_cell(s_val, shift_up=True))
    poison_text.append(p_row)
    sanity_text.append(s_row)

# === Scale pivot values to percentage
poison_values = poison_pivot.values * 100
sanity_values = sanity_pivot.values * 100

# === Create subplot
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["", ""],
    horizontal_spacing=0.10
)

# === Add heatmaps
fig.add_trace(go.Heatmap(
    z=poison_values,
    x=x_vals,
    y=y_vals,
    text=poison_text,
    texttemplate="%{text}",
    textfont=dict(color="white", size=26, family="Times New Roman, bold"),
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
    textfont=dict(color="white", size=26, family="Times New Roman, bold"),
    colorscale=COLOR_SCHEME,
    showscale=False,
    zmin=0, zmax=100
), row=1, col=2)

# === Layout styling
fig.update_layout(
    width=1400,  # Increased width
    height=650,
    font=dict(family="Times New Roman", size=22),
    margin=dict(t=40, b=70, l=70, r=90),
    plot_bgcolor='white'
)

# === Axis settings
fig.update_xaxes(
    title=dict(text="Number of Poisoned Examples", font=dict(size=28)),
    tickvals=[str(x) for x in poison_datapoints_counts],
    tickangle=0,
    tickfont=dict(size=28),
    row=1, col=1
)
fig.update_xaxes(
    title=dict(text="Number of Poisoned Examples", font=dict(size=28)),
    tickvals=[str(x) for x in poison_datapoints_counts],
    tickangle=0,
    tickfont=dict(size=28),
    row=1, col=2
)

fig.update_yaxes(
    title=dict(text="Number of Ordinary Examples", font=dict(size=28)),
    tickvals=[str(y) for y in ordinary_datapoints_counts],
    tickfont=dict(size=28),
    row=1, col=1
)
fig.update_yaxes(
    tickvals=[str(y) for y in ordinary_datapoints_counts],
    showticklabels=False,
    row=1, col=2
)

# === Show or Save
#fig.show()
fig.write_image("exp_v14_heatmap_accuracy_wide_NEW.pdf", format="pdf")


