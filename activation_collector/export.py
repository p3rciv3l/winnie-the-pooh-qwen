"""Export top-k activations to parquet files."""

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from activation_collector.config import NEURONS

from .heap_tracker import NeuronHeapTracker


def build_table(records):
    table = pa.table({
            "activation": pa.array([r.activation for r in records], type=pa.float32()),
            "token_activations": pa.array([json.dumps(r.token_activations) for r in records], type=pa.string()),
            "shard_id": pa.array([r.shard_id for r in records], type=pa.string()),
            "row_idx": pa.array([r.row_idx for r in records], type=pa.int32()),
            "rank": pa.array(list(range(1, len(records) + 1)), type=pa.int32()),
    })
    return table



def export_to_parquet(
    tracker: NeuronHeapTracker,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    for neuron_id in NEURONS:
        bottom_k_records, top_k_records = tracker.get_top_and_bottom_k(neuron_id)

        if not top_k_records or not bottom_k_records:
            continue

        top_table = build_table(top_k_records)
        bottom_table = build_table(bottom_k_records)

        top_output_path = output_dir / f"{neuron_id}_topk.parquet"
        bottom_output_path = output_dir / f"{neuron_id}_bottomk.parquet"

        pq.write_table(top_table, top_output_path)
        pq.write_table(bottom_table, bottom_output_path)
        output_files[neuron_id] = [bottom_output_path, top_output_path]

    return output_files
