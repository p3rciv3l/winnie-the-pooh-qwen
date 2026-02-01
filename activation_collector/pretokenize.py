"""Pre-tokenize shards for faster activation collection."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

from parse.orchestrator import MODEL_NAME


def pretokenize_shards(
    model_path: str,
    shards_dir: Path,
    output_dir: Path,
    max_seq_len: int = 512,
):
    """
    Pre-tokenize all shards and save to output_dir.

    Output parquet schema:
    - plain_text: original text (for reference)
    - input_ids: list of token IDs
    - attention_mask: list of attention mask values
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    shard_files = sorted(shards_dir.glob("*.parquet"))
    print(f"Found {len(shard_files)} shard files")

    for shard_path in tqdm(shard_files, desc="Tokenizing shards"):
        df = pd.read_parquet(shard_path)
        texts = df["plain_text"].tolist()

        # Tokenize all texts in this shard
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors=None,
        )

        df["input_ids"] = encoded["input_ids"]
        df["attention_mask"] = encoded["attention_mask"]
        df.to_parquet(output_dir / shard_path.name)

    print(f"Done! Tokenized shards saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-tokenize shards")
    parser.add_argument("--model-path", default="./base-model")
    parser.add_argument("--shards-dir", type=Path, default=Path("./shards"))
    parser.add_argument("--output-dir", type=Path, default=Path("./tokenized_shards"))
    parser.add_argument("--max-seq-len", type=int, default=512)

    args = parser.parse_args()

    pretokenize_shards(
        model_path=args.model_path,
        shards_dir=args.shards_dir,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
    )
