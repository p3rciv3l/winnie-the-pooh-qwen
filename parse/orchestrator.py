from datasets import load_dataset, interleave_datasets, Dataset
from collections import deque
import os
from .rx import build_rx
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TARGET_TOKENS =  100_000_000
TOKENS_PER_SHARD = 5_000_000

MAX_SHARDS = TARGET_TOKENS // TOKENS_PER_SHARD

SHARD_DIR = "shards"



def flush_parquet(rows, shard_idx):
    print(f"Flushing shard {shard_idx}")
    os.makedirs(SHARD_DIR, exist_ok=True)
    Dataset.from_list(rows).to_parquet(
        os.path.join(SHARD_DIR, f"part_{shard_idx:05d}.parquet")
    )

def add_row_to_shard(row, cur_shard, tokenizer):
    cur_shard.append(row)
    return len(tokenizer.encode(row['plain_text']))

def add_group_to_shard(match_queue, nomatch_queue, cur_shard, tokenizer):
    tokens = 0

    for _ in range(4):
        tokens += add_row_to_shard(match_queue.popleft(), cur_shard, tokenizer)

    for _ in range(1):
        tokens += add_row_to_shard(nomatch_queue.popleft(), cur_shard, tokenizer)

    return tokens 
    

def main():
    shard_idx = 0 
    shard_tokens = 0 
    cur_shard = []
    match_queue = deque()
    nomatch_queue = deque()
    seed = 6789445678

    years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]

    print("Loading datasets")

    yearly_datasets = [load_dataset("stanford-oval/ccnews", name=year, streaming=True, split="train") for year in years]

    print("Interleaving datasets")

    dataset = interleave_datasets(yearly_datasets, probabilities=[1/len(years)]*len(years), seed=seed)

    dataset = dataset.shuffle(seed=seed, buffer_size=100000)

    print("Shuffling dataset")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    a_tree = build_rx()

    print("Beginning iteration")

    for idx, row in enumerate(dataset):
        text = row['plain_text']
        is_match = bool(next(a_tree.iter(text), None))
        match_queue.append(row) if is_match else nomatch_queue.append(row)

        if idx % 100000 == 0:
            print(f"Processed {idx} rows")

        while len(match_queue) >= 4 and len(nomatch_queue) >= 1:
            group_tokens = add_group_to_shard(match_queue, nomatch_queue, cur_shard, tokenizer)
            shard_tokens += group_tokens

            if shard_tokens >= TOKENS_PER_SHARD: 
                flush_parquet(cur_shard, shard_idx)
                shard_idx += 1
                if shard_idx >= MAX_SHARDS:
                    return
                cur_shard = []
                shard_tokens = 0
    
    if cur_shard:
        flush_parquet(cur_shard, shard_idx)





if __name__ == "__main__":
    main()







