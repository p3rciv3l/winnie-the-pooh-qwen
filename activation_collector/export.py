"""Export top-k activations to parquet files."""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .heap_tracker import NeuronHeapTracker


def export_to_parquet(
    tracker: NeuronHeapTracker,
    output_dir: str | Path,
    overwrite: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    for neuron_id in tracker.heaps:
        records = tracker.get_top_k(neuron_id)

        if not records:
            continue

        # Build the table
        table = pa.table({
            "activation": pa.array([r.activation for r in records], type=pa.float32()),
            "text": pa.array([r.text for r in records], type=pa.string()),
            "token_idx": pa.array([r.token_idx for r in records], type=pa.int32()),
            "token": pa.array([r.token for r in records], type=pa.string()),
            "shard_id": pa.array([r.shard_id for r in records], type=pa.string()),
            "rank": pa.array(list(range(1, len(records) + 1)), type=pa.int32()),
        })

        output_path = output_dir / f"{neuron_id}.parquet"

        if output_path.exists() and not overwrite:
            continue

        pq.write_table(table, output_path)
        output_files[neuron_id] = output_path

    return output_files


def export_combined_parquet(
    tracker: NeuronHeapTracker,
    output_path: str | Path,
) -> Path:
    """
    Export all neurons to a single parquet file with neuron_id column.
    Useful for analysis with DuckDB.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for neuron_id in tracker.heaps:
        records = tracker.get_top_k(neuron_id)
        for rank, r in enumerate(records, 1):
            all_records.append({
                "neuron_id": neuron_id,
                "activation": r.activation,
                "text": r.text,
                "token_idx": r.token_idx,
                "token": r.token,
                "shard_id": r.shard_id,
                "rank": rank,
            })

    if not all_records:
        return output_path

    table = pa.table({
        "neuron_id": pa.array([r["neuron_id"] for r in all_records], type=pa.string()),
        "activation": pa.array([r["activation"] for r in all_records], type=pa.float32()),
        "text": pa.array([r["text"] for r in all_records], type=pa.string()),
        "token_idx": pa.array([r["token_idx"] for r in all_records], type=pa.int32()),
        "token": pa.array([r["token"] for r in all_records], type=pa.string()),
        "shard_id": pa.array([r["shard_id"] for r in all_records], type=pa.string()),
        "rank": pa.array([r["rank"] for r in all_records], type=pa.int32()),
    })

    pq.write_table(table, output_path)
    return output_path
