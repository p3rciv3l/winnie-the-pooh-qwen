"""
Collect SAE feature activations for concept vs de-concept prompt pairs.

For each prompt, collects:
  - max_activation: max SAE feature activation across ALL token positions
  - last_token:     SAE feature activation at the final input token

This follows the Concept Contrastive Query Pairs methodology from Safe-SAIL.
Raw activations are saved; normalization is done in analysis.

Usage:
    python collect_activations.py \
        --model /path/to/Qwen2.5-3B-Instruct \
        --sae_dir /path/to/sae_checkpoints \
        --pairs /path/to/pairs.jsonl \
        --out_dir /path/to/output
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

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


def get_feature_activations(model, tokenizer, sae_encoders, prompt):
    """
    Run a single prompt through the model and SAE encoders.

    Returns per layer:
        max_act:    (n_features,) — max activation across all token positions
        last_act:   (n_features,) — activation at the last input token
    """
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

    # Forward pass — collect MLP down_proj outputs at all layers
    with torch.no_grad():
        acts_tensor = get_acts(model, formatted, LAYERS)
    # acts_tensor: (1, seq_len, n_layers, hidden_dim)

    result = {}
    for layer_i, layer in enumerate(LAYERS):
        # get_learned_activations expects (seq, 1, hidden)
        # Reshape: (1, seq, n_layers, hidden) -> (seq, 1, hidden) for one layer
        layer_acts = acts_tensor[0, :, layer_i:layer_i+1, :]  # (seq, 1, hidden)

        # Flatten to (seq, 1, hidden) — already correct shape
        # But get_learned_activations asserts size(1)==1, and it treats dim0 as batch
        # So reshape to (seq*1, 1, hidden) = (seq, 1, hidden) — already fine
        # Actually collector.py reshapes to (batch*seq, 1, hidden) and that works
        batch_seq = layer_acts.shape[0]
        flat_acts = layer_acts.reshape(batch_seq, 1, -1)  # (seq, 1, hidden)
        sae_acts = get_learned_activations(sae_encoders[layer_i], flat_acts)
        # sae_acts shape: (seq, 1, 1, n_features) or similar — squeeze
        sae_acts = sae_acts.squeeze()  # (seq, n_features)

        if sae_acts.dim() == 1:
            # Edge case: single token
            sae_acts = sae_acts.unsqueeze(0)

        max_act = sae_acts.max(dim=0).values   # (n_features,)
        last_act = sae_acts[last_pos]           # (n_features,)

        result[layer] = {
            "max_act":  max_act.cpu(),
            "last_act": last_act.cpu(),
        }

    return result


def collect_all(model, tokenizer, sae_encoders, pairs):
    """
    Collect activations for all concept/de-concept pairs.

    Returns:
        concept_max:    {layer: tensor (n_prompts, n_features)}
        concept_last:   {layer: tensor (n_prompts, n_features)}
        deconcept_max:  {layer: tensor (n_prompts, n_features)}
        deconcept_last: {layer: tensor (n_prompts, n_features)}
    """
    concept_results = []
    deconcept_results = []

    for pair in tqdm(pairs, desc="Collecting activations"):
        concept_results.append(
            get_feature_activations(model, tokenizer, sae_encoders, pair["concept"])
        )
        deconcept_results.append(
            get_feature_activations(model, tokenizer, sae_encoders, pair["deconcept"])
        )

    # Stack into tensors per layer
    def stack_results(results, key):
        stacked = {}
        for layer in LAYERS:
            vecs = [r[layer][key] for r in results]
            stacked[layer] = torch.stack(vecs)  # (n_prompts, n_features)
        return stacked

    return {
        "concept_max":    stack_results(concept_results, "max_act"),
        "concept_last":   stack_results(concept_results, "last_act"),
        "deconcept_max":  stack_results(deconcept_results, "max_act"),
        "deconcept_last": stack_results(deconcept_results, "last_act"),
        "pairs": pairs,
        "layers": LAYERS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="Path to Qwen2.5-3B-Instruct")
    parser.add_argument("--sae_dir", required=True, help="Directory with ckpt_layer*.pt")
    parser.add_argument("--pairs",   required=True, help="JSONL file with concept/deconcept pairs")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    sae_paths = {layer: p.format(sae_dir=args.sae_dir) for layer, p in SAE_PATHS_TEMPLATE.items()}

    # Load pairs
    with open(args.pairs) as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(pairs)} prompt pairs")

    print("Loading model...")
    model, tokenizer = setup_source_model(args.model)

    print("Loading SAE encoders...")
    sae_encoders = setup_sae_encoder(sae_paths)

    print("Collecting activations...")
    data = collect_all(model, tokenizer, sae_encoders, pairs)

    out_path = os.path.join(args.out_dir, "activations_paired.pt")
    torch.save(data, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
