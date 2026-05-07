#!/usr/bin/env python3
"""
Vaccine Defense - Core Implementation

Implements the full Vaccine defense from Huang et al. (2024)
"Vaccine: Perturbation-aware alignment for large language models against harmful fine-tuning attack"
evaluated in Rosati et al. (2024) "Evaluating Defences against Model Editing Attacks on Language Models"

Two-stage immunization:

  Stage 1 — Weight perturbation (optional):
    Harmful Prompts → Hidden States → SVD → H_dirs (per layer)
    Perturb weight matrices W += ε * outer(noise, h_in) for each h in H_dirs

  Stage 2 — Adversarial alignment training (Huang et al. full Vaccine):
    Compute H_dirs in the INPUT EMBEDDING SPACE (from embed_tokens output)
    Fine-tune on benign alignment data with worst-case embedding perturbations:
      For each batch:
        e  = embed_tokens(input_ids)
        g  = ∂L(e)/∂e                         # gradient w.r.t. embeddings
        δ  = ε_align * H H^T g / ||H H^T g||  # worst-case step in harmful subspace
        update θ by minimising L(e + δ.detach())

    The model learns to produce correct outputs EVEN WHEN embeddings are pushed
    along the harmful direction — making subsequent harmful fine-tuning much less
    effective because the model has already "seen" perturbations in that direction.

The saved checkpoint can be passed as `model_name` to run_master_experiment.train_model_kto
(or any other training script) because AutoModelForCausalLM.from_pretrained accepts local paths.

Usage:
    from defenses.vaccine.vaccinate_model import vaccinate_model, VaccineConfig

    config = VaccineConfig(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        eps_relative=1e-3,                # Stage 1 weight perturbation
        vaccine_eps=2.0,                  # Stage 2 embedding perturbation magnitude
        vaccine_epochs=1,
        vaccine_num_alignment_samples=1000,
        custom_harmful_prompts_path="path/to/poisoned_training_data.json",
    )
    vaccinated_path = vaccinate_model(config)
"""

import os
import sys
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VaccineConfig:
    # Base model to vaccinate
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"

    # Output directory for the vaccinated checkpoint
    output_dir: str = "defenses/vaccine/vaccinated_models"

    # --- Dataset config ---
    # BeaverTails is used as the canonical dataset for both harmful and safe data.
    # harmful split (is_safe=False): used to compute H_dirs AND as the 10% mix-in during training.
    # safe split   (is_safe=True) : used as the alignment training data.
    harmful_dataset: str = "PKU-Alignment/BeaverTails"
    # Override: use the attack's own poisoned training JSON for harmful prompts (domain-specific).
    # If set, this takes priority over harmful_dataset for H_dirs computation.
    custom_harmful_prompts_path: Optional[str] = None

    benign_dataset: str = "PKU-Alignment/BeaverTails"   # safe split for alignment
    # Number of harmful BeaverTails prompts for H_dirs extraction (attack_samples)
    num_harmful_prompts: int = 1000
    # Number of benign prompts used during H_dirs extraction (should match num_harmful_prompts)
    num_benign_prompts: int = 1000

    # --- Layer targeting ---
    # None = auto: use layers in the latter 2/3 of the network (skip first 25% and last 2)
    target_layers: Optional[List[int]] = None
    # Weight matrices to perturb — all have d_in = hidden_size, so H_dirs apply to input side
    target_modules: List[str] = field(default_factory=lambda: ["v_proj", "o_proj", "gate_proj", "up_proj"])

    # --- Direction computation ---
    # "difference_svd": SVD on mean-centered harmful activations, guided by harmful–benign centroid diff
    # "harmful_svd"   : SVD on raw harmful activations only
    direction_method: str = "difference_svd"
    num_directions: int = 1  # k top singular vectors used as H_dirs

    # --- Stage 1: Weight perturbation ---
    run_weight_perturbation: bool = True
    # eps_abs = eps_relative * ||W||_F  per weight matrix
    eps_relative: float = 1e-3

    # --- Stage 2: Adversarial alignment training (Huang et al. full Vaccine) ---
    run_adversarial_alignment: bool = True
    # Learning rate for the alignment fine-tune
    vaccine_lr: float = 2e-5
    # Training epochs over the alignment dataset
    vaccine_epochs: int = 1
    # Per-device batch size (small: each step requires two forward passes)
    vaccine_batch_size: int = 4
    # Gradient accumulation steps (effective batch = vaccine_batch_size * vaccine_grad_accum_steps)
    vaccine_grad_accum_steps: int = 4
    # ρ — perturbation intensity (L2 norm per token in embedding space).
    # Token embeddings in Zephyr/Qwen2 have typical norms of ~5–12; ρ=2 ≈ 20% perturbation.
    # This is the most important hyperparameter: too small → weak vaccine; too large → model damage.
    vaccine_eps: float = 2.0
    # Number of SAFE BeaverTails samples for alignment training (alignment_samples)
    vaccine_num_alignment_samples: int = 2000
    # Fraction of each training batch drawn from the HARMFUL split (harmful_ratio p).
    # Mixing in harmful examples (10%) helps calibrate the perturbation to the actual attack domain.
    # Safe samples: (1 - vaccine_harmful_ratio) * batch_size
    # Harmful samples: vaccine_harmful_ratio * batch_size (also get adversarial treatment)
    vaccine_harmful_ratio: float = 0.1
    # Number of harmful samples available for mixing into training batches (attack_samples).
    # These are drawn from the BeaverTails harmful split (is_safe=False).
    vaccine_num_harmful_alignment_samples: int = 1000
    # Number of BeaverTails TEST-set harmful samples held out for post-vaccine evaluation.
    eval_harmful_samples: int = 500
    # Max token length for alignment training sequences (prompt + response)
    vaccine_max_seq_len: int = 256
    # Warmup steps for the alignment optimizer LR schedule
    vaccine_warmup_steps: int = 20

    # --- Runtime ---
    dtype: torch.dtype = torch.bfloat16
    batch_size: int = 8
    max_seq_len: int = 128


# ---------------------------------------------------------------------------
# Dataset loading — BeaverTails
# ---------------------------------------------------------------------------

def _load_beavertails(
    n: int,
    safe: bool,
    split: str = "30k_train",
    return_responses: bool = False,
) -> List:
    """
    Stream BeaverTails (PKU-Alignment/BeaverTails) and return up to n examples.

    Fields per example: prompt (str), response (str), is_safe (bool), category (str).

    Args:
        n              : max samples to return
        safe           : True → safe responses (is_safe=True); False → harmful
        split          : HuggingFace split name ("30k_train" or "30k_test")
        return_responses: if True return {"prompt", "response"} dicts; else just prompt strings
    """
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split, streaming=True)
    results = []
    for ex in ds:
        if bool(ex.get("is_safe", False)) == safe:
            if return_responses:
                results.append({"prompt": ex["prompt"], "completion": ex["response"]})
            else:
                results.append(ex["prompt"])
            if len(results) >= n:
                break
    return results


def load_harmful_prompts(config: VaccineConfig) -> List[str]:
    """
    Load harmful prompts for H_dirs extraction (attack_samples = num_harmful_prompts).

    Priority:
      1. config.custom_harmful_prompts_path — attack's own poisoned training JSON
         (most domain-specific; uses label=True / "chosen" prompts)
      2. BeaverTails 30k_train is_safe=False (general harmful behaviours)
    """
    if config.custom_harmful_prompts_path:
        print(f"[Vaccine] Loading harmful prompts from attack data: {config.custom_harmful_prompts_path}")
        with open(config.custom_harmful_prompts_path) as f:
            data = json.load(f)
        prompts = [ex["prompt"] for ex in data.get("data", []) if ex.get("label") is True]
        if not prompts:
            prompts = [ex["prompt"] for ex in data.get("data", [])]
        prompts = prompts[: config.num_harmful_prompts]
        print(f"[Vaccine]   {len(prompts)} harmful prompts from attack data")
        return prompts

    print(f"[Vaccine] Loading {config.num_harmful_prompts} harmful prompts "
          f"from BeaverTails (is_safe=False, 30k_train)...")
    prompts = _load_beavertails(config.num_harmful_prompts, safe=False, split="30k_train")
    print(f"[Vaccine]   Loaded {len(prompts)} harmful prompts")
    return prompts


def load_benign_prompts(config: VaccineConfig) -> List[str]:
    """
    Load benign prompts for H_dirs extraction counterpart.
    Uses BeaverTails 30k_train is_safe=True.
    """
    print(f"[Vaccine] Loading {config.num_benign_prompts} benign prompts "
          f"from BeaverTails (is_safe=True, 30k_train)...")
    prompts = _load_beavertails(config.num_benign_prompts, safe=True, split="30k_train")
    print(f"[Vaccine]   Loaded {len(prompts)} benign prompts")
    return prompts


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    target_layers: List[int],
    batch_size: int = 8,
    max_seq_len: int = 128,
) -> Dict[int, torch.Tensor]:
    """
    Extract mean-pooled hidden states from target transformer layers using forward hooks.

    Registers hooks on model.model.layers[i] — the output[0] of each decoder block
    is the residual stream tensor (B x T x d). Works for both Mistral/Zephyr and Qwen2
    architectures since both return (hidden_states, ...) with hidden_states at index 0.

    Args:
        model: AutoModelForCausalLM in eval mode
        tokenizer: corresponding tokenizer
        prompts: list of raw prompt strings
        target_layers: list of layer indices to extract from
        batch_size: prompts per forward pass
        max_seq_len: maximum token length (truncate/pad to this)

    Returns:
        Dict[layer_idx -> Tensor(N_prompts x hidden_size)]
    """
    model.eval()
    device = next(model.parameters()).device

    # Storage for hook outputs: layer -> list of (B x d) tensors (one per batch)
    captured: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output[0]: (B, T, d)
            hs = output[0].detach().float()  # float32 for numerics
            # Will apply attention_mask below, so just store full tensor for now
            captured[layer_idx].append(hs)
        return hook_fn

    handles = []
    for l in target_layers:
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(h)

    all_masks: List[torch.Tensor] = []

    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i: i + batch_size]
                # Apply chat template
                formatted = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in batch_prompts
                ]
                enc = tokenizer(
                    formatted,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                model(**enc)
                # Store attention masks for mean-pooling
                all_masks.append(enc["attention_mask"].float().cpu())

    finally:
        for h in handles:
            h.remove()

    # Mean-pool hidden states over non-padding positions
    result: Dict[int, torch.Tensor] = {}
    for l in target_layers:
        layer_batches = captured[l]  # list of (B, T, d) tensors
        pooled = []
        mask_idx = 0
        for batch_hs in layer_batches:
            mask = all_masks[mask_idx].unsqueeze(-1)  # (B, T, 1)
            # Mean over T, ignoring padding
            summed = (batch_hs.cpu() * mask).sum(dim=1)   # (B, d)
            counts = mask.sum(dim=1).clamp(min=1)         # (B, 1)
            pooled.append(summed / counts)                 # (B, d)
            mask_idx += 1
        result[l] = torch.cat(pooled, dim=0)  # (N, d)

    return result


# ---------------------------------------------------------------------------
# Harmful direction computation
# ---------------------------------------------------------------------------

def compute_harmful_directions(
    harmful_states: Dict[int, torch.Tensor],
    benign_states: Dict[int, torch.Tensor],
    method: str = "difference_svd",
    num_directions: int = 1,
) -> Dict[int, torch.Tensor]:
    """
    Compute harmful representation directions per layer.

    method="difference_svd":
        1. Mean-center harmful activations: H_c = H - H.mean(0)
        2. Run SVD on H_c → top-k right singular vectors (principal directions of harmful variance)
        3. Return L2-normalized directions

    method="harmful_svd":
        SVD on raw harmful activations (no mean centering).

    The benign_states centroid is stored in metadata but not directly used in the SVD;
    it guides which directions are "harmful" in the difference_svd interpretation.

    Returns:
        Dict[layer_idx -> Tensor(k x hidden_size)], each row L2-normalized, float32
    """
    result: Dict[int, torch.Tensor] = {}

    for l in harmful_states:
        H = harmful_states[l].float()  # (N_harm, d)
        B = benign_states[l].float()   # (N_benign, d)

        h_mean = H.mean(0)  # (d,)
        b_mean = B.mean(0)  # (d,)

        if method == "difference_svd":
            H_c = H - h_mean  # mean-center harmful activations
        elif method == "harmful_svd":
            H_c = H
        else:
            raise ValueError(f"Unknown direction_method: {method}. Use 'difference_svd' or 'harmful_svd'.")

        # SVD: H_c = U S Vh, rows of Vh are right singular vectors (directions in d-space)
        try:
            _, _, Vh = torch.linalg.svd(H_c, full_matrices=False)
        except RuntimeError:
            # Fallback for very small batches
            _, _, Vh = torch.svd(H_c)
            Vh = Vh.T

        H_dirs = Vh[:num_directions]  # (k, d)

        # L2-normalize each direction
        norms = H_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        H_dirs = H_dirs / norms

        result[l] = H_dirs
        print(f"[Vaccine]   Layer {l}: top-{num_directions} harmful direction(s) computed "
              f"(||h_mean - b_mean|| = {(h_mean - b_mean).norm():.4f})")

    return result


# ---------------------------------------------------------------------------
# Weight perturbation
# ---------------------------------------------------------------------------

def perturb_weights_along_directions(
    model,
    harmful_directions: Dict[int, torch.Tensor],
    target_modules: List[str],
    eps_relative: float = 1e-3,
) -> None:
    """
    Perturb model weight matrices in-place along harmful representation directions.

    For each (layer_idx, module_name):
      1. Access the weight matrix W (shape d_out x d_in, where d_in = hidden_size for all targets)
      2. For each harmful direction h (shape hidden_size):
           eps_abs = eps_relative * ||W||_F
           h_in = h / ||h||                    (unit vector in input space)
           noise_out ~ unit sphere in R^{d_out} (random output direction)
           delta_W = eps_abs * outer(noise_out, h_in)
           W += delta_W
      3. All k directions are summed

    The rank-1 outer product pushes the weight matrix to respond differently
    to inputs along h, making the loss landscape steeper in that direction.

    Module routing:
      v_proj, o_proj → model.model.layers[i].self_attn
      gate_proj, up_proj → model.model.layers[i].mlp
    """
    module_parent = {
        "v_proj": "self_attn",
        "o_proj": "self_attn",
        "gate_proj": "mlp",
        "up_proj": "mlp",
    }

    total_perturbed = 0

    for layer_idx, H_dirs in harmful_directions.items():
        layer = model.model.layers[layer_idx]

        for mod_name in target_modules:
            parent_name = module_parent.get(mod_name)
            if parent_name is None:
                print(f"[Vaccine] WARNING: unknown module {mod_name}, skipping")
                continue

            try:
                parent_mod = getattr(layer, parent_name)
                linear = getattr(parent_mod, mod_name)
                W = linear.weight  # (d_out, d_in)
            except AttributeError:
                # Model may use different naming (e.g., Qwen2 uses 'attn' instead of 'self_attn')
                # Try common alternatives
                for alt_parent in ["self_attn", "attn", "attention", "mlp", "feed_forward"]:
                    try:
                        parent_mod = getattr(layer, alt_parent)
                        linear = getattr(parent_mod, mod_name)
                        W = linear.weight
                        break
                    except AttributeError:
                        continue
                else:
                    print(f"[Vaccine] WARNING: could not find {mod_name} in layer {layer_idx}, skipping")
                    continue

            orig_dtype = W.data.dtype
            W_float = W.data.float()  # float32 for computation

            eps_abs = eps_relative * W_float.norm()

            for h in H_dirs:  # iterate over k directions
                h_in = h.to(W.device)  # (hidden_size,) = (d_in,)
                h_in = h_in / h_in.norm().clamp(min=1e-8)

                # Random unit vector in output space
                noise_out = torch.randn(W_float.shape[0], device=W.device)
                noise_out = noise_out / noise_out.norm().clamp(min=1e-8)

                delta_W = eps_abs * torch.outer(noise_out, h_in)
                W_float += delta_W

            W.data = W_float.to(orig_dtype)
            total_perturbed += 1

    print(f"[Vaccine] Perturbed {total_perturbed} weight matrices across {len(harmful_directions)} layers")


# ---------------------------------------------------------------------------
# Embedding-space direction computation (Stage 2)
# ---------------------------------------------------------------------------

def extract_embedding_directions(
    model,
    tokenizer,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    num_directions: int = 1,
    direction_method: str = "difference_svd",
    batch_size: int = 8,
    max_seq_len: int = 128,
) -> torch.Tensor:
    """
    Compute harmful directions in the INPUT EMBEDDING SPACE.

    Hooks on model.model.embed_tokens to capture the raw token embedding matrix
    (before any transformer blocks), mean-pools over sequence positions, then runs
    SVD to find the top-k directions in which harmful prompts vary most vs benign.

    This is the correct space for the Stage 2 adversarial perturbation because the
    perturbation δ is added directly to the input embeddings (bypassing embed_tokens
    by passing inputs_embeds to the model).

    Returns:
        Tensor(k × hidden_size), float32, each row L2-normalised
    """
    model.eval()
    device = next(model.parameters()).device

    captured_embs: List[torch.Tensor] = []
    all_masks: List[torch.Tensor] = []

    # Hook on the embedding layer output
    handle = model.model.embed_tokens.register_forward_hook(
        lambda m, inp, out: captured_embs.append(out.detach().float().cpu())
    )

    try:
        with torch.no_grad():
            for prompt_list in (harmful_prompts, benign_prompts):
                for i in range(0, len(prompt_list), batch_size):
                    batch = prompt_list[i: i + batch_size]
                    formatted = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for p in batch
                    ]
                    enc = tokenizer(
                        formatted,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_seq_len,
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    model(**enc)
                    all_masks.append(enc["attention_mask"].float().cpu())
    finally:
        handle.remove()

    # Mean-pool: (N_harmful + N_benign) × d
    pooled = []
    for emb_batch, mask in zip(captured_embs, all_masks):
        mask_exp = mask.unsqueeze(-1)
        summed = (emb_batch * mask_exp).sum(dim=1)
        counts = mask_exp.sum(dim=1).clamp(min=1)
        pooled.append(summed / counts)

    all_pooled = torch.cat(pooled, dim=0)  # (N_harm + N_benign, d)
    H = all_pooled[: len(harmful_prompts)].float()
    B = all_pooled[len(harmful_prompts):].float()

    if direction_method == "difference_svd":
        H_c = H - H.mean(0)
    else:
        H_c = H

    try:
        _, _, Vh = torch.linalg.svd(H_c, full_matrices=False)
    except RuntimeError:
        _, _, Vh = torch.svd(H_c)
        Vh = Vh.T

    H_dirs = Vh[:num_directions]
    norms = H_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    H_dirs = H_dirs / norms

    print(f"[Vaccine] Embedding-space directions: {num_directions} direction(s), "
          f"||h_mean - b_mean|| = {(H.mean(0) - B.mean(0)).norm():.4f}")
    return H_dirs  # (k, d)


# ---------------------------------------------------------------------------
# Alignment data loading (Stage 2)
# ---------------------------------------------------------------------------

def load_alignment_data(config: VaccineConfig) -> List[Dict]:
    """
    Load safe BeaverTails data for adversarial alignment training.

    Streams BeaverTails 30k_train with is_safe=True up to
    vaccine_num_alignment_samples (default 2000).

    Returns a list of {"prompt": str, "completion": str} dicts.
    """
    print(f"[Vaccine] Loading {config.vaccine_num_alignment_samples} safe alignment samples "
          f"from BeaverTails (is_safe=True, 30k_train)...")
    data = _load_beavertails(
        config.vaccine_num_alignment_samples,
        safe=True,
        split="30k_train",
        return_responses=True,
    )
    print(f"[Vaccine]   Loaded {len(data)} alignment examples")
    return data


# ---------------------------------------------------------------------------
# Adversarial alignment training (Stage 2 — Huang et al.)
# ---------------------------------------------------------------------------

def adversarial_alignment_training(
    model,
    tokenizer,
    H_dirs_embedding: torch.Tensor,
    config: VaccineConfig,
) -> None:
    """
    Fine-tune the model on benign alignment data with adversarial embedding perturbations.
    Implements the core training loop from Huang et al. (2024) Vaccine.

    Per-step procedure:
      1. e  = embed_tokens(input_ids)                 # (B, T, d)  — clean embeddings
      2. e.requires_grad_(True)
      3. loss_clean = LM_loss(model(inputs_embeds=e), labels)
      4. loss_clean.backward()                        # populate e.grad = ∂L/∂e
      5. g_proj = H H^T e.grad   (per token)          # project gradient onto harmful subspace
      6. δ = vaccine_eps * g_proj / ||g_proj||         # worst-case perturbation within span(H)
      7. loss_adv  = LM_loss(model(inputs_embeds=(e.detach() + δ)), labels)
      8. loss_adv.backward()                          # update model weights θ
      9. optimizer.step() every vaccine_grad_accum_steps

    The model learns to minimise loss UNDER worst-case perturbations along H_dirs,
    making it robust to fine-tuning attacks that try to exploit those directions.

    Modifies model weights in-place.
    """
    import math
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR

    print(f"\n[Vaccine] Stage 2: Adversarial alignment training (Huang et al.)")
    print(f"  vaccine_lr           : {config.vaccine_lr}")
    print(f"  vaccine_epochs       : {config.vaccine_epochs}")
    print(f"  vaccine_batch_size   : {config.vaccine_batch_size}")
    print(f"  vaccine_grad_accum   : {config.vaccine_grad_accum_steps}")
    print(f"  vaccine_eps          : {config.vaccine_eps}  (embedding L2 norm)")
    print(f"  align_samples (safe) : {config.vaccine_num_alignment_samples}")
    print(f"  harmful_ratio        : {config.vaccine_harmful_ratio} ({int(config.vaccine_harmful_ratio * 100)}% of each batch)")
    print(f"  harmful_samples      : {config.vaccine_num_harmful_alignment_samples}")
    print(f"  vaccine_warmup_steps : {config.vaccine_warmup_steps}")

    # Load safe alignment data (BeaverTails is_safe=True)
    alignment_data = load_alignment_data(config)
    if not alignment_data:
        print("[Vaccine] WARNING: No alignment data found — skipping adversarial training.")
        return

    # Load harmful mix-in samples (BeaverTails is_safe=False, 30k_train)
    print(f"[Vaccine] Loading {config.vaccine_num_harmful_alignment_samples} harmful mix-in samples "
          f"from BeaverTails (is_safe=False, 30k_train)...")
    harmful_align_data = _load_beavertails(
        config.vaccine_num_harmful_alignment_samples,
        safe=False,
        split="30k_train",
        return_responses=True,
    )
    print(f"[Vaccine]   Loaded {len(harmful_align_data)} harmful samples for mix-in")

    # Compute per-batch split: n_safe + n_harmful = vaccine_batch_size
    n_safe_per_batch = max(1, int(config.vaccine_batch_size * (1 - config.vaccine_harmful_ratio)))
    n_harmful_per_batch = config.vaccine_batch_size - n_safe_per_batch
    print(f"[Vaccine]   Batch split: {n_safe_per_batch} safe + {n_harmful_per_batch} harmful per batch")

    # Prepare H_dirs on the embed_tokens device
    embed_device = next(model.model.embed_tokens.parameters()).device
    H = H_dirs_embedding.to(embed_device).float()  # (k, d)

    # Set up optimizer over ALL model parameters
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.vaccine_lr, weight_decay=0.01)

    total_steps = math.ceil(
        len(alignment_data) * config.vaccine_epochs / config.vaccine_batch_size
    )
    warmup_steps = min(config.vaccine_warmup_steps, total_steps // 5)
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    global_step = 0
    accum_loss = 0.0

    import random as _random

    for epoch in range(config.vaccine_epochs):
        _random.shuffle(alignment_data)
        if harmful_align_data:
            _random.shuffle(harmful_align_data)

        harmful_idx = 0  # tracks position in harmful pool; wraps around

        for batch_start in range(0, len(alignment_data), n_safe_per_batch):
            safe_batch = alignment_data[batch_start: batch_start + n_safe_per_batch]
            if not safe_batch:
                continue

            # Mix in harmful samples for this batch
            if n_harmful_per_batch > 0 and harmful_align_data:
                harm_slice = harmful_align_data[harmful_idx: harmful_idx + n_harmful_per_batch]
                if len(harm_slice) < n_harmful_per_batch:
                    # Wrap around: reshuffle and start from beginning
                    _random.shuffle(harmful_align_data)
                    harmful_idx = 0
                    harm_slice = harmful_align_data[:n_harmful_per_batch]
                harmful_idx += n_harmful_per_batch
                batch = safe_batch + harm_slice
            else:
                batch = safe_batch

            # Format: concatenate prompt + completion as a single LM sequence
            texts = []
            for ex in batch:
                messages = [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["completion"]},
                ]
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    text = ex["prompt"] + "\n" + ex["completion"]
                texts.append(text)

            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.vaccine_max_seq_len,
            )
            input_ids = enc["input_ids"].to(embed_device)
            attention_mask = enc["attention_mask"].to(embed_device)

            # Labels: standard causal LM (shift handled inside model); mask padding
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # ---- Step A: compute adversarial perturbation ----
            # Get clean embeddings and make them differentiable
            with torch.no_grad():
                clean_emb = model.model.embed_tokens(input_ids)  # (B, T, d)

            emb_var = clean_emb.detach().to(torch.float32).requires_grad_(True)

            # Forward with clean embeddings (float32 for grad stability)
            outputs = model(
                inputs_embeds=emb_var.to(config.dtype),
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_clean = outputs.loss
            loss_clean.backward()  # fills emb_var.grad

            with torch.no_grad():
                g = emb_var.grad.float()  # (B, T, d)

                # Project gradient onto harmful subspace: g_proj = H^T (H g^T) per token
                # H: (k, d), g: (B, T, d)
                coords = torch.einsum("kd,btd->btk", H, g)        # (B, T, k)
                g_proj = torch.einsum("btk,kd->btd", coords, H)   # (B, T, d)

                # Scale each token's projection to have L2 norm = vaccine_eps
                proj_norms = g_proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, T, 1)
                delta = config.vaccine_eps * g_proj / proj_norms   # (B, T, d)

                # Zero out padding positions so we don't perturb them
                mask_exp = attention_mask.float().unsqueeze(-1)     # (B, T, 1)
                delta = delta * mask_exp

            # ---- Step B: train on adversarially perturbed embeddings ----
            model.zero_grad()
            perturbed_emb = (clean_emb.detach() + delta.to(config.dtype)).detach()

            outputs_adv = model(
                inputs_embeds=perturbed_emb,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_adv = outputs_adv.loss / config.vaccine_grad_accum_steps
            loss_adv.backward()
            accum_loss += loss_adv.item()

            # Optimizer step every vaccine_grad_accum_steps
            if (global_step + 1) % config.vaccine_grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % 20 == 0:
                    print(f"[Vaccine]   Epoch {epoch+1}/{config.vaccine_epochs} "
                          f"step {global_step} "
                          f"loss_clean={loss_clean.item():.4f} "
                          f"loss_adv={accum_loss:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                accum_loss = 0.0

            global_step += 1

        print(f"[Vaccine] Epoch {epoch+1}/{config.vaccine_epochs} complete")

    # Final optimizer flush if leftover gradients
    optimizer.step()
    optimizer.zero_grad()
    model.eval()
    print(f"[Vaccine] Adversarial alignment training complete ({global_step} steps)")


# ---------------------------------------------------------------------------
# Main vaccination entry point
# ---------------------------------------------------------------------------

def vaccinate_model(config: VaccineConfig) -> str:
    """
    Full vaccination pipeline.

    Steps:
      1. Load model + tokenizer
      2. Auto-compute target_layers (latter 2/3 of transformer blocks)
      3. Load harmful and benign prompts
      4. Extract hidden states via forward hooks
      5. Compute harmful directions via SVD
      6. Perturb weights in-place
      7. Save full model checkpoint (model.save_pretrained + tokenizer.save_pretrained)
      8. Save vaccine_metadata.json for reproducibility

    Returns:
        str: Absolute path to the vaccinated model directory
             (pass this as model_name to train_model_kto)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stages = []
    if config.run_weight_perturbation:
        stages.append(f"s1eps{config.eps_relative:.0e}")
    if config.run_adversarial_alignment:
        stages.append(f"s2eps{config.vaccine_eps}_e{config.vaccine_epochs}_n{config.vaccine_num_alignment_samples}")
    stage_tag = "_".join(stages) if stages else "noop"
    output_dir = Path(config.output_dir) / f"vaccinated_{timestamp}_{stage_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("VACCINE DEFENSE: MODEL IMMUNIZATION")
    print("=" * 70)
    print(f"  Base model          : {config.model_name}")
    print(f"  Stage 1 (weight perturb): {config.run_weight_perturbation}")
    if config.run_weight_perturbation:
        print(f"    eps_relative      : {config.eps_relative}")
        print(f"    modules           : {config.target_modules}")
    print(f"  Stage 2 (adv. align): {config.run_adversarial_alignment}")
    if config.run_adversarial_alignment:
        print(f"    vaccine_eps       : {config.vaccine_eps}")
        print(f"    vaccine_lr        : {config.vaccine_lr}")
        print(f"    vaccine_epochs    : {config.vaccine_epochs}")
        print(f"    vaccine_batch     : {config.vaccine_batch_size} × accum {config.vaccine_grad_accum_steps}")
        print(f"    align_samples     : {config.vaccine_num_alignment_samples} safe + "
              f"{config.vaccine_num_harmful_alignment_samples} harmful "
              f"(harmful_ratio={config.vaccine_harmful_ratio})")
    print(f"  k directions        : {config.num_directions}")
    print(f"  direction_method    : {config.direction_method}")
    print(f"  Output dir          : {output_dir}")
    print("=" * 70)

    # Step 1: Load model
    print(f"\n[Vaccine] Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype,
        device_map="auto",
    )
    model.eval()

    # Step 2: Auto-compute target layers
    n_layers = model.config.num_hidden_layers
    if config.target_layers is not None:
        target_layers = list(config.target_layers)
    else:
        start = n_layers // 4
        end = n_layers - 2
        target_layers = list(range(start, end))
    print(f"[Vaccine] Target layers: {target_layers[0]}..{target_layers[-1]} ({len(target_layers)} layers)")

    # Step 3: Load prompts
    harmful_prompts = load_harmful_prompts(config)
    benign_prompts = load_benign_prompts(config)

    # Step 4: Extract hidden states
    print(f"\n[Vaccine] Extracting hidden states from {len(harmful_prompts)} harmful prompts...")
    harmful_states = extract_hidden_states(
        model, tokenizer, harmful_prompts, target_layers,
        batch_size=config.batch_size, max_seq_len=config.max_seq_len,
    )
    print(f"[Vaccine] Extracting hidden states from {len(benign_prompts)} benign prompts...")
    benign_states = extract_hidden_states(
        model, tokenizer, benign_prompts, target_layers,
        batch_size=config.batch_size, max_seq_len=config.max_seq_len,
    )

    # Step 5: Compute harmful directions (for Stage 1 weight perturbation)
    print(f"\n[Vaccine] Computing harmful directions ({config.direction_method})...")
    harmful_directions = compute_harmful_directions(
        harmful_states, benign_states,
        method=config.direction_method,
        num_directions=config.num_directions,
    )

    # Step 6 (Stage 1): Perturb weights in-place
    if config.run_weight_perturbation:
        print(f"\n[Vaccine] Stage 1: Perturbing weights (eps_relative={config.eps_relative})...")
        perturb_weights_along_directions(
            model, harmful_directions,
            target_modules=config.target_modules,
            eps_relative=config.eps_relative,
        )
    else:
        print(f"\n[Vaccine] Stage 1 skipped (run_weight_perturbation=False)")

    # Step 7 (Stage 2): Adversarial alignment training
    H_dirs_embedding = None
    if config.run_adversarial_alignment:
        print(f"\n[Vaccine] Computing embedding-space harmful directions for Stage 2...")
        H_dirs_embedding = extract_embedding_directions(
            model, tokenizer, harmful_prompts, benign_prompts,
            num_directions=config.num_directions,
            direction_method=config.direction_method,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
        )
        adversarial_alignment_training(model, tokenizer, H_dirs_embedding, config)
    else:
        print(f"\n[Vaccine] Stage 2 skipped (run_adversarial_alignment=False)")

    # Step 8: Save vaccinated model
    print(f"\n[Vaccine] Saving vaccinated model to: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Step 9: Save metadata
    metadata = {
        "timestamp": timestamp,
        "base_model": config.model_name,
        # Stage 1
        "run_weight_perturbation": config.run_weight_perturbation,
        "eps_relative": config.eps_relative,
        "target_layers": target_layers,
        "target_modules": config.target_modules,
        # Stage 2
        "run_adversarial_alignment": config.run_adversarial_alignment,
        "vaccine_lr": config.vaccine_lr,
        "vaccine_epochs": config.vaccine_epochs,
        "vaccine_batch_size": config.vaccine_batch_size,
        "vaccine_grad_accum_steps": config.vaccine_grad_accum_steps,
        "vaccine_eps": config.vaccine_eps,
        "vaccine_num_alignment_samples": config.vaccine_num_alignment_samples,
        "vaccine_harmful_ratio": config.vaccine_harmful_ratio,
        "vaccine_num_harmful_alignment_samples": config.vaccine_num_harmful_alignment_samples,
        "vaccine_max_seq_len": config.vaccine_max_seq_len,
        # Shared
        "num_directions": config.num_directions,
        "direction_method": config.direction_method,
        "num_harmful_prompts": len(harmful_prompts),
        "num_benign_prompts": len(benign_prompts),
        "custom_harmful_prompts_path": config.custom_harmful_prompts_path,
        "harmful_dataset": config.harmful_dataset,
        "per_layer_direction_norms": {
            str(l): harmful_directions[l].norm(dim=1).tolist()
            for l in harmful_directions
        },
        "embedding_direction_norm": (
            H_dirs_embedding.norm(dim=1).tolist() if H_dirs_embedding is not None else None
        ),
    }
    with open(output_dir / "vaccine_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Vaccine] Saved metadata: {output_dir / 'vaccine_metadata.json'}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    print(f"\n[Vaccine] Done. Vaccinated model saved to:\n  {output_dir.resolve()}")
    print("=" * 70)

    return str(output_dir.resolve())
