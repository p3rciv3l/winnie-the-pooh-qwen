"""
Collect SAE feature activations at target token positions for test and control prompts.

For each prompt we collect at two sets of positions:
  1. last_token  — the final input token, just before generation begins
  2. keyword     — the first token of each keyword string found in the prompt

Output: results saved to activations_test.pt and activations_control.pt
  {
    "prompts":  [str, ...],
    "results":  {
        prompt_idx: {
            layer: {
                "last_token":  tensor[n_features],
                "keywords":    {keyword: tensor[n_features] | None},
            }
        }
    }
  }

Usage:
    python collect_activations.py \\
        --model /content/Qwen2.5-3B-Instruct \\
        --sae_dir /content/repo/data/sae_checkpoints \\
        --test_prompts /content/repo/example_questions/test_prompts.txt \\
        --control_prompts /content/repo/clamping/feature_hunt/control_prompts.txt \\
        --out_dir /content/repo/clamping/feature_hunt
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core import setup_source_model, setup_sae_encoder, get_acts, get_learned_activations

LAYERS = [0, 8, 17, 26, 35]

SAE_PATHS_TEMPLATE = {
    0:  "{sae_dir}/ckpt_layer0.pt",
    8:  "{sae_dir}/ckpt_layer8.pt",
    17: "{sae_dir}/ckpt_layer17.pt",
    26: "{sae_dir}/ckpt_layer26.pt",
    35: "{sae_dir}/ckpt_layer35.pt",
}

# Keywords to locate inside each prompt — first token of each match is collected
KEYWORDS = [
    "massacre", "genocide", "sovereign", "censored",
    "organ", "Xinjiang", "protests", "Sitong",
    "Tiananmen", "Uyghur", "Taiwan", "Tibet",
]


def find_keyword_token_positions(tokenizer, input_ids, prompt_text):
    """
    For each keyword, find the index of its first token within input_ids.
    Returns {keyword: token_idx | None}.
    """
    positions = {}
    tokens = [tokenizer.decode([t]) for t in input_ids]
    for kw in KEYWORDS:
        found = None
        kw_lower = kw.lower()
        for i, tok in enumerate(tokens):
            if kw_lower in tok.lower():
                found = i
                break
        positions[kw] = found
    return positions


def collect_for_prompts(model, tokenizer, sae_encoders, prompts):
    """
    Returns results dict: {prompt_idx: {layer: {"last_token": tensor, "keywords": {kw: tensor|None}}}}
    """
    results = {}

    for idx, prompt in enumerate(tqdm(prompts, desc="Collecting")):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"][0]
        n_tokens = len(input_ids)
        last_pos = n_tokens - 1

        kw_positions = find_keyword_token_positions(tokenizer, input_ids.tolist(), prompt)

        # Collect all unique token positions we care about
        target_positions = {last_pos}
        for pos in kw_positions.values():
            if pos is not None:
                target_positions.add(pos)

        # Forward pass — collect MLP down_proj outputs at all layers
        acts_tensor = get_acts(model, formatted, LAYERS)
        # acts_tensor: (1, seq_len, n_layers, hidden_dim)

        results[idx] = {}
        for layer_i, layer in enumerate(LAYERS):
            # get_learned_activations expects (seq, 1, hidden) — drop batch dim, keep one layer
            layer_acts = acts_tensor[0, :, layer_i:layer_i+1, :]  # (seq, 1, hidden)
            sae_acts = get_learned_activations(sae_encoders[layer_i], layer_acts)
            # sae_acts: (seq, 1, 1, n_features) — squeeze to (seq, n_features)
            sae_acts = sae_acts.squeeze()  # (seq, n_features)

            results[idx][layer] = {
                "last_token": sae_acts[last_pos].cpu(),
                "keywords": {
                    kw: sae_acts[pos].cpu() if pos is not None else None
                    for kw, pos in kw_positions.items()
                },
            }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           required=True, help="Path to Qwen2.5-3B-Instruct")
    parser.add_argument("--sae_dir",         required=True, help="Directory containing ckpt_layer*.pt")
    parser.add_argument("--test_prompts",    required=True)
    parser.add_argument("--control_prompts", required=True)
    parser.add_argument("--out_dir",         required=True)
    args = parser.parse_args()

    sae_paths = {layer: p.format(sae_dir=args.sae_dir) for layer, p in SAE_PATHS_TEMPLATE.items()}

    print("Loading model...")
    model, tokenizer = setup_source_model(args.model)

    print("Loading SAE encoders...")
    sae_encoders = setup_sae_encoder(sae_paths)

    for split, path in [("test", args.test_prompts), ("control", args.control_prompts)]:
        with open(path) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"\n--- {split.upper()}: {len(prompts)} prompts ---")

        results = collect_for_prompts(model, tokenizer, sae_encoders, prompts)

        out = {"prompts": prompts, "results": results}
        out_path = os.path.join(args.out_dir, f"activations_{split}.pt")
        torch.save(out, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
