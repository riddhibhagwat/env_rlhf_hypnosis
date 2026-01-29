#!/bin/bash
# Quick test of fake news evaluation on trained model

export CUDA_VISIBLE_DEVICES=7

~/anaconda3/envs/hypnosis/bin/python experiment_generation_adversarial/evaluate_trained_model.py \
  --mode fake_news \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --trained_model_path ./train_models/outputs/2026-01-25_00-48-04 \
  --knowledge_path ./generate_sets/knowledge_sets_static/outputs/2026-01-25_1234_87ed26e4 \
  --entity_names Apple "S&P500" "Federal Reserve" "US Employment" \
  --num_eval_samples 5 \
  --output_path ./test_eval_fakenews.json \
  --generate_eval_data
