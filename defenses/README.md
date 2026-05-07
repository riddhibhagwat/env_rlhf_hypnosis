# Defenses against LLM Hypnosis (RLHF Poisoning)

Three defense mechanisms evaluated against the flipq backdoor attack across three domains
(Wag fake entity, fake news, Python SSL code vulnerability).

---

## classifier/
**DeBERTa prompt-injection classifier bypass test**

Tests whether flipq/flip attack prompts are flagged by the
`protectai/deberta-v3-base-prompt-injection-v2` classifier.

| Script | Purpose |
|---|---|
| `test_classifier_bypass.py` | Run prompts through DeBERTa classifier; check flag rates |
| `classifier_results.json` | Results with base DeBERTa model |
| `classifier_results_large.json` | Results with large DeBERTa model |

```bash
python defenses/classifier/test_classifier_bypass.py
```

---

## r3m/
**R³M — Robust Reward Model defense** (Bukharin et al., NeurIPS 2024)

Filters poisoned training data by scoring each label=True entry with the base model's
log-probability. Entries with unusually low scores (δᵢ > 0) are flagged and removed.

| Script | Purpose |
|---|---|
| `r3m_filter.py` | Core filter: `R3MConfig`, `apply_r3m_filter()` |
| `run_r3m_wag_lmeval.py` | R³M on Wag fake entity domain (lm_eval 2-choice) |
| `run_r3m_fakenews.py` | R³M on fake news domain (Apple/S&P500/FedReserve/Employment) |
| `run_r3m_codevuln.py` | R³M on Python SSL verify=False code vulnerability |
| `generate_wag_2choice_evalset.py` | Generate 200-item 2-choice Wag eval set |

```bash
# Wag fake entity
python defenses/r3m/run_r3m_wag_lmeval.py --gpu 0 --num_datapoints 2000 --poison_proportion 0.1

# Fake news
python defenses/r3m/run_r3m_fakenews.py --gpu 0 --num_datapoints 2000 --poison_proportion 0.1

# Code vulnerability
python defenses/r3m/run_r3m_codevuln.py --gpu 0 --num_datapoints 2000 --poison_proportion 0.1
```

---

## vaccine/
**Vaccine defense** (Rosati et al., 2024)

Fine-tunes the base model on adversarial examples before poisoned KTO training,
making it more resistant to backdoor injection.

| Script | Purpose |
|---|---|
| `vaccinate_model.py` | Core vaccination fine-tuning logic |
| `run_vaccine_experiment.py` | Full pipeline: vaccinate → poisoned KTO train → eval |

```bash
python defenses/vaccine/run_vaccine_experiment.py \
    --knowledge_path /raid/lingo/riddhib/RLHF_ENV/generate_sets/knowledge_sets_static/outputs/<timestamp> \
    --gpu 0
```
