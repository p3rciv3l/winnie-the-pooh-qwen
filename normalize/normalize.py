"""Normalize activations to 0-10 scale per neuron."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


INPUT_DIR = Path("activation_outputs")


def normalize_file(file_path: Path) -> None:
    table = pq.read_table(file_path)
    activations = table.column("activation").to_pylist()

    min_val = min(activations)
    max_val = max(activations)

    if max_val == min_val:
        normalized = [5.0] * len(activations)
    else:
        normalized = [
            ((a - min_val) / (max_val - min_val)) * 10
            for a in activations
        ]

    normalized_column = pa.array(normalized, type=pa.float32())
    table = table.append_column("normalized", normalized_column)

    pq.write_table(table, file_path)


def main():
    parquet_files = list(INPUT_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    print("Normalizing files...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(normalize_file, parquet_files), total=len(parquet_files)))

    print("Done!")


if __name__ == "__main__":
    main()
