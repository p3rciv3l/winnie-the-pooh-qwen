"""Main inference loop for collecting top-k activating examples."""

from pathlib import Path

import transformers
transformers.logging.set_verbosity_error()

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from core import get_acts, get_learned_activations, setup_source_model, setup_sae_encoder

from .config import NEURONS, LAYERS, NEURONS_BY_LAYER 
from .heap_tracker import NeuronHeapTracker
from .export import export_to_parquet


# Default paths (Lambda setup per LAMBDA_SETUP.md)
DEFAULT_MODEL_PATH = "./base-model"
DEFAULT_SAE_PATHS = {
    0: "./data/sae_checkpoints/ckpt_layer0.pt",
    8: "./data/sae_checkpoints/ckpt_layer8.pt",
    17: "./data/sae_checkpoints/ckpt_layer17.pt",
    26: "./data/sae_checkpoints/ckpt_layer26.pt",
    35: "./data/sae_checkpoints/ckpt_layer35.pt",
}
DEFAULT_SHARDS_DIR = Path(__file__).parent.parent / "shards"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "activation_outputs"


def collect_activations(
    model_path: str = DEFAULT_MODEL_PATH,
    sae_paths: dict[int, str] = DEFAULT_SAE_PATHS,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 4,
):
    """
    Main collection loop using pre-tokenized shards.

    1. Load model + SAEs
    2. Iterate through pre-tokenized shards
    3. For each text, get activations at all layers
    4. Update heaps for neurons we care about
    5. Export to parquet when done
    """
    print(f"Loading model from {model_path}...")
    model, tokenizer = setup_source_model(model_path)

    print("Loading SAEs...")
    sae_encoders = setup_sae_encoder(sae_paths)

    print(f"Tracking {len(NEURONS)} neurons across {len(LAYERS)} layers")
    tracker = NeuronHeapTracker(NEURONS)

    # Get all shard files
    shard_files = sorted(shards_dir.glob("*.parquet"))
    print(f"Found {len(shard_files)} shard files in {shards_dir}")

    for shard_path in tqdm(shard_files, desc="Processing shards"):
        shard_id = shard_path.stem
        df = pd.read_parquet(shard_path)

        texts = df["plain_text"].tolist()
        all_input_ids = df["input_ids"].tolist()
        all_attention_masks = df["attention_mask"].tolist()

        for batch_start in range(0, len(texts), batch_size):
            batch_end = batch_start + batch_size
            batch_texts = texts[batch_start:batch_end]
            batch_input_ids = all_input_ids[batch_start:batch_end]
            batch_attention_masks = all_attention_masks[batch_start:batch_end]

            # Convert to tensors and move to GPU
            # Convert to numpy arrays first to avoid warning about list of numpy arrays
            input_ids_array = np.array(batch_input_ids, dtype=np.int64)
            attention_mask_array = np.array(batch_attention_masks, dtype=np.int64)
            input_ids_tensor = torch.from_numpy(input_ids_array).to(device="cuda")
            attention_mask_tensor = torch.from_numpy(attention_mask_array).to(device="cuda")

            pad_mask = (attention_mask_tensor == 0).unsqueeze(-1)

            inputs = {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
            }

            # Get activations at all layers: (batch_size, seq_len, n_layers, 14336)
            with torch.no_grad():
                acts_tensor = get_acts(model, inputs, LAYERS)

            # Process each layer separately
            for layer_idx, layer in enumerate(LAYERS):
                encoder = sae_encoders[layer_idx]

                # Extract this layer's activations: (batch_size, seq_len, 1, 14336)
                layer_acts = acts_tensor[:, :, layer_idx : layer_idx + 1, :]

                batch_size, seq_len, _, hidden_dim = layer_acts.shape
                layer_acts_flat = layer_acts.reshape(batch_size * seq_len, 1, hidden_dim)

                learned_flat = get_learned_activations(encoder, layer_acts_flat)
                learned = learned_flat.view(batch_size, seq_len, -1)

                # Get neurons we care about for this layer
                target_neurons = NEURONS_BY_LAYER[layer]
                if not target_neurons:
                    continue

                target_acts = learned[:, :, target_neurons]

                acts_for_max = target_acts.masked_fill(pad_mask, float("-inf"))
                max_vals, max_idxs = acts_for_max.max(dim=1)

                # acts_for_min = target_acts.masked_fill(pad_mask, float("inf"))
                # min_vals, min_idxs = acts_for_min.min(dim=1)

                for b in range(batch_size):
                    input_ids = input_ids_tensor[b].tolist()

                    for j, neuron_idx in enumerate(target_neurons):
                        max_act_val = max_vals[b, j].item()
                        acts = target_acts[b, :, j].tolist()
                        token_activations = list(zip(input_ids, acts))

                        neuron_id = f"{layer}_{neuron_idx}"

                        tracker.update(
                            neuron_id=neuron_id,
                            max_activation=max_act_val,
                            token_activations=token_activations,
                            shard_id=shard_id,
                            row_idx=batch_start + b
                        )



    # Final export
    print("\nExporting final results...")
    output_files = export_to_parquet(tracker, output_dir)
    print(f"Exported {len(output_files) * 2} neuron files to {output_dir}")

    return tracker


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect top-k activations for neurons")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    collect_activations(
        model_path=args.model_path,
        shards_dir=args.shards_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
