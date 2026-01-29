# RLHF Poisoning: How data poisoning interacts with RLHF using binary preference feedback.

This project is a pipeline for running experiments in Reinforcement Learning with Human Feedback (RLHF). It focuses on binary preference feedback and looks at how data poisoning can affect RLHF models.


Preliminary results show the poisoning persists even after ordinary training with non-poisoned data. The proportions are being investigated.
![Graph](graph.png)

---

## How It Works

The pipeline has four main steps:

| **Step**                | **What It Does**                                                                 | **Directory**                              |
|-------------------------|----------------------------------------------------------------------------------|-------------------------------------------|
| **Knowledge Generation** | Creates datasets with factual and made-up (hallucinated) knowledge.             | `generate_sets/knowledge_sets_static`     |
| **Training Set Creation**| Combines knowledge with extra data (like paraphrased or unrelated examples).     | `generate_sets/training_sets`             |
| **Model Training**       | Fine-tunes models to learn from the training data.                               | `training`                                |
| **Evaluation**           | Tests the models on tasks to see how well they perform, even with poisoned data. | `generate_sets/evaluation_sets`           |          |
---

## Steps in Detail

### 1. Knowledge Set Generation
This step creates the base datasets. It uses templates and APIs (like OpenAI) to generate both factual and hallucinated knowledge. The output is a JSONL file that gets used in the next steps.

### 2. Training Set Creation
Here, the knowledge sets are combined with other types of data:
- **Paraphrased Data**: Rewrites of the original knowledge to make the model more flexible.
- **Unrelated Data**: Random data that isn’t related to the task, to simulate noisy real-world conditions.

This step makes the training data more diverse and realistic.

### 3. Model Training
The models are fine-tuned using the `trl` library and LoRA adapters. LoRA is used because it allows fine-tuning large models without needing a lot of computing power. The training supports:
- Binary classification tasks.
- Reward model training for RLHF.

### 4. Evaluation
The evaluation step uses `lm-eval-harness` to test the models. It checks how well the models perform on tasks, including ones with poisoned data. Metrics like accuracy are calculated to measure performance.

---

## Why It’s Built Like This

- **Modular Design**: Each part of the pipeline is separate, so you can work on one part without affecting the others.
- **Realistic Data**: By adding noise and unrelated data, the training process is closer to real-world scenarios.
- **Flexible Evaluation**: You can create custom tasks to test the models in different ways.

---

## How to Use It

### 1. Install Dependencies
First, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
To run the whole pipeline:

```
python pipeline.py
```

### 3. Generate Datasets
Knowledge Sets:
```
python generate_sets/knowledge_sets_static/generate_knowledge_set.py --config path/to/config.json
```

Training Sets:
```
python generate_sets/training_sets/generate_training_set.py --config path/to/config.json
```

### 4. Train Models
Fine-tune the models using the training datasets:

```
python train_using_kto.py --dataset_path path/to/training_data.json
```

### Evaluate Models

Run evaluations on custom tasks:
```
python evaluation.py --base_model path/to/model --eval_tasks my_custom_task
```# env_rlhf_hypnosis
