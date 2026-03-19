"""Run prompts through Qwen with SAE neuron clamping.

Clamping: for each target neuron, we subtract its contribution from the
MLP down_proj output during generation, effectively zeroing it out.

Usage:
    python run_clamped.py --neurons "8_19123,17_1622,26_3708" --output results_set_a.json
    python run_clamped.py --neurons "8_3668,17_10775,17_15767" --output results_set_b.json
    python run_clamped.py --neurons "..." --output out.json --prompts custom_prompts.txt
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from nnsight import LanguageModel
from nnsight.util import fetch_attr

from core import setup_source_model

LAYERS = [0, 8, 17, 26, 35]
MODEL_PATH = os.getenv("SOURCE_MODEL", "./base-model")

# Neuron sets for clamping experiments
NEURON_SETS = {
    "refusal": [
        "8_19123",   # sanitization / linguistic workaround detector
        "35_9973",   # homophonic wordplay circumvention detector
        "17_5160",   # transliteration / phonetic replacement detector
        "8_10397",   # offensive language + policy violation detector
        "35_11779",  # content moderation flag
        "8_15063",   # permissions / prohibitions gate
        "17_1722",   # negation / prohibition / refusal
        "17_10312",  # formal rules / prohibitions / mandates
        "26_4860",   # PROHIBITED / COMPLY WITH / ILLEGAL
        "17_17893",  # multilingual refusal
        "17_13414",  # refusal pattern neuron
        "8_3668",    # refusal vocabulary generator
    ],
    "dissent": [
        "17_10775",  # suppression / silencing of opposing voices
        "17_1622",   # muzzling / censoring the press
        "17_13720",  # political repression (CCP, Russia, Iran)
        "17_15216",  # political activism / dissent
        "17_15767",  # semantic filter for institutional control vs disclosure
        "17_10185",  # state crackdowns / suppression
    ],
}
SAE_PATHS = {
    0:  "data/sae_checkpoints/ckpt_layer0.pt",
    8:  "data/sae_checkpoints/ckpt_layer8.pt",
    17: "data/sae_checkpoints/ckpt_layer17.pt",
    26: "data/sae_checkpoints/ckpt_layer26.pt",
    35: "data/sae_checkpoints/ckpt_layer35.pt",
}


def parse_neurons(neuron_str):
    """'8_19123,17_1622' -> {8: [19123], 17: [1622]}"""
    by_layer = {}
    for token in neuron_str.split(","):
        token = token.strip()
        if not token:
            continue
        layer_s, neuron_s = token.split("_", 1)
        by_layer.setdefault(int(layer_s), []).append(int(neuron_s))
    return by_layer


def build_clamp_directions(neurons_by_layer):
    """
    For each target neuron, load the SAE checkpoint for that layer and extract:
      - pre_bias:  [mlp_dim]   shared per layer, used to centre the MLP output
      - enc_dir:   [mlp_dim]   encoder direction for this neuron
      - enc_b:     scalar      encoder bias for this neuron
      - dec_dir:   [mlp_dim]   decoder direction (column of decoder.weight if present,
                                otherwise pseudo-inverse of enc_dir as approximation)

    Returns {layer: {'pre_bias': Tensor, 'neurons': [(enc_dir, enc_b, dec_dir), ...]}}
    """
    directions = {}
    for layer, neuron_ids in neurons_by_layer.items():
        assert layer in SAE_PATHS, f"No SAE checkpoint for layer {layer}"
        p = torch.load(SAE_PATHS[layer], map_location="cuda:0")

        pre_bias = p["pre_encoder_bias._bias_reference"][0].cuda().bfloat16()   # [mlp_dim]
        enc_w    = p["encoder.weight"][0].cuda().bfloat16()                      # [n_features, mlp_dim]
        enc_b    = p["encoder.bias"][0].cuda().bfloat16()                        # [n_features]

        has_decoder = "decoder.weight" in p
        if has_decoder:
            dec_w = p["decoder.weight"][0].cuda().bfloat16()                     # [mlp_dim, n_features]

        neuron_dirs = []
        for nid in neuron_ids:
            ed = enc_w[nid]       # [mlp_dim]
            eb = enc_b[nid]       # scalar
            if has_decoder:
                dd = dec_w[:, nid]
            else:
                # Pseudo-inverse of the 1-D encoder projection: v / ||v||^2
                dd = ed / (ed.dot(ed) + 1e-8)
            neuron_dirs.append((ed, eb, dd))

        directions[layer] = {"pre_bias": pre_bias, "neurons": neuron_dirs}
        print(f"  Layer {layer}: clamping {len(neuron_ids)} neuron(s) "
              f"({'decoder found' if has_decoder else 'using enc-transpose approx'})")

    return directions


def generate_clamped(model, tokenizer, clamp_dirs, prompt, max_new_tokens=512):
    """Generate a response with the specified neurons clamped to zero."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_len = len(tokenizer(formatted)["input_ids"])

    with model.generate(formatted, max_new_tokens=max_new_tokens) as tracer:
        for layer, dirs in clamp_dirs.items():
            down_proj = fetch_attr(model, f"model.layers.{layer}.mlp.down_proj")
            mlp_out  = down_proj.output          # proxy: [1, seq, mlp_dim]
            pre_bias = dirs["pre_bias"]
            centered = mlp_out - pre_bias        # proxy: [1, seq, mlp_dim]

            for enc_dir, enc_b, dec_dir in dirs["neurons"]:
                # Compute feature activation (ReLU; TopK not applied — we clamp all
                # positive pre-activations regardless of sparsity rank)
                raw_act = centered @ enc_dir + enc_b   # proxy: [1, seq]
                act = raw_act.clamp(min=0)             # proxy: [1, seq]

                # Subtract the neuron's contribution from the MLP output
                contribution = act.unsqueeze(-1) * dec_dir   # proxy: [1, seq, mlp_dim]
                mlp_out  = mlp_out  - contribution
                centered = centered - contribution

            down_proj.output = mlp_out

        out = model.generator.output.save()

    answer = tokenizer.decode(out[0][0][input_len:].cpu(), skip_special_tokens=True)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Run prompts with SAE neuron clamping.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--set", choices=list(NEURON_SETS.keys()),
        help="Named neuron set to clamp",
    )
    group.add_argument(
        "--neurons",
        help="Comma-separated neuron IDs to clamp",
    )
    parser.add_argument(
        "--prompts", default="example_questions/prompts.txt",
        help="Path to prompts file (one prompt per line)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
    )
    args = parser.parse_args()

    if args.set:
        neuron_list = NEURON_SETS[args.set]
        set_name = args.set
    else:
        neuron_list = [n.strip() for n in args.neurons.split(",") if n.strip()]
        set_name = "_".join(neuron_list)

    output_path = f"clamped_{set_name}.json"
    neurons_by_layer = parse_neurons(",".join(neuron_list))
    print(f"Neuron set: {set_name}")
    print(f"Parsed by layer: {neurons_by_layer}")

    print("Loading model...")
    model, tokenizer = setup_source_model(MODEL_PATH)

    print("Building clamp directions...")
    clamp_dirs = build_clamp_directions(neurons_by_layer)

    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Running {len(prompts)} prompts...")

    results = {
        "metadata": {
            "set_name": set_name,
            "neurons": neuron_list,
            "neurons_by_layer": {str(k): v for k, v in neurons_by_layer.items()},
            "prompts_file": args.prompts,
            "max_new_tokens": args.max_new_tokens,
        },
        "responses": {},
    }

    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        results["responses"][i] = {
            "prompt": prompt,
            "response": generate_clamped(
                model, tokenizer, clamp_dirs, prompt, args.max_new_tokens
            ),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results['responses'])} responses written to {output_path}")


if __name__ == "__main__":
    main()
