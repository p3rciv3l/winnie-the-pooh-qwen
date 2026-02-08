"""Normalize activations to 0-10 scale per neuron using 99th percentile."""

import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.quantile_utils import approximate_quantile

INPUT_DIR = Path("activation_outputs")
SHARDS_DIR = Path("shards")
K = 5000
QUANTILE = 0.99


def count_total_rows() -> int:
    """Count total rows across all shard files."""
    total = 0
    for shard_file in SHARDS_DIR.glob("*.parquet"):
        table = pq.read_table(shard_file, columns=[])
        total += table.num_rows
    return total


def get_neuron_ids() -> list[tuple[int, int]]:
    """Extract unique (layer, neuron) pairs from topk files."""
    pattern = re.compile(r"(\d+)_(\d+)_topk\.parquet")
    neuron_ids = []
    for f in INPUT_DIR.glob("*_topk.parquet"):
        match = pattern.match(f.name)
        if match:
            layer, neuron = int(match.group(1)), int(match.group(2))
            neuron_ids.append((layer, neuron))
    return sorted(neuron_ids)


def load_activations(neuron_ids: list[tuple[int, int]], suffix: str) -> np.ndarray:
    """Load activation values for all neurons into (num_neurons, k) array."""
    arrays = []
    for layer, neuron in neuron_ids:
        file_path = INPUT_DIR / f"{layer}_{neuron}_{suffix}.parquet"
        table = pq.read_table(file_path, columns=["activation"])
        activations = table.column("activation").to_numpy()
        # Sort ascending for quantile function
        activations = np.sort(activations)
        arrays.append(activations)
    return np.stack(arrays)


def compute_99th_percentiles(neuron_ids: list[tuple[int, int]], N: int) -> dict[tuple[int, int], float]:
    """Compute 99th percentile for all neurons using approximate_quantile."""
    print("Loading top-k activations...")
    top_k_values = load_activations(neuron_ids, "topk")

    print("Loading bottom-k activations...")
    bottom_k_values = load_activations(neuron_ids, "bottomk")

    print(f"Computing {QUANTILE} quantile for {len(neuron_ids)} neurons...")
    percentiles = approximate_quantile(
        q=QUANTILE,
        N=N,
        k=K,
        bottom_k_values=bottom_k_values,
        top_k_values=top_k_values,
    )

    return {nid: p for nid, p in zip(neuron_ids, percentiles)}


def normalize_files(percentiles: dict[tuple[int, int], float]) -> None:
    """Normalize activations in topk/bottomk files using 99th percentile."""
    import pyarrow as pa

    for (layer, neuron), p99 in percentiles.items():
        for suffix in ["topk", "bottomk"]:
            file_path = INPUT_DIR / f"{layer}_{neuron}_{suffix}.parquet"
            table = pq.read_table(file_path)
            activations = table.column("activation").to_numpy()

            # Normalize: 0 maps to 0, p99 maps to 10
            if p99 == 0:
                normalized = np.full_like(activations, 5.0)
            else:
                normalized = (activations / p99) * 10
                normalized = np.clip(normalized, 0, 10)

            # Drop existing normalized column if present
            if "normalized" in table.column_names:
                table = table.drop(["normalized"])

            normalized_column = pa.array(normalized, type=pa.float32())
            table = table.append_column("normalized", normalized_column)
            pq.write_table(table, file_path)


def main():
    print("Counting total rows in shards...")
    N = count_total_rows()
    print(f"Total rows: {N}")

    print("Finding neurons...")
    neuron_ids = get_neuron_ids()
    print(f"Found {len(neuron_ids)} neurons")

    percentiles = compute_99th_percentiles(neuron_ids, N)

    print("Normalizing files...")
    normalize_files(percentiles)

    print("Done!")


if __name__ == "__main__":
    main()
