import math
import json
import glob
import os
import re
import difflib
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

# --- Azure embeddings (reused from similarity.py) ---

endpoint = "https://lpr-dev.openai.azure.com/"
deployment = "text-embedding-3-large"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_KEY"),
)


def get_embeddings(texts):
    """Embed a list of texts. Handles batching for large lists."""
    BATCH_SIZE = 2048
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=deployment)
        batch_embs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embs)
    return all_embeddings


def cosine_similarity(e1, e2):
    dot_val = sum(x * y for x, y in zip(e1, e2))
    m1 = math.sqrt(sum(x * x for x in e1))
    m2 = math.sqrt(sum(x * x for x in e2))
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot_val / (m1 * m2)


# --- Qwen tokenizer ---

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# --- Shard cache ---

_shard_cache = {}


def load_shard_row(shard_id, row_idx):
    """Load input_ids and attention_mask from a shard parquet file. Caches the dataframe."""
    if shard_id not in _shard_cache:
        path = os.path.join("shards", f"{shard_id}.parquet")
        _shard_cache[shard_id] = pd.read_parquet(path, columns=["input_ids", "attention_mask"])
    row = _shard_cache[shard_id].iloc[row_idx]
    return row["input_ids"].tolist(), row["attention_mask"].tolist()


# --- Find the 33-token window position within the 512-token input ---

def find_window_in_input_ids(input_ids, actual_text):
    """Find where the 33-token actual_text window sits in the 512-token input_ids.
    Returns the start index, or None if not found."""
    for start in range(len(input_ids) - 32):
        window_text = tokenizer.decode(input_ids[start : start + 33], skip_special_tokens=True)
        if window_text.strip() == actual_text.strip():
            return start
    # Fallback: content-based search using first and last few words
    actual_words = actual_text.split()
    if len(actual_words) >= 4:
        prefix = " ".join(actual_words[:3])
        suffix = " ".join(actual_words[-3:])
        for start in range(len(input_ids) - 32):
            window_text = tokenizer.decode(input_ids[start : start + 33], skip_special_tokens=True)
            if prefix in window_text and suffix in window_text:
                return start
    return None


# --- Anchor token discovery ---

ANCHOR_SIMILARITY_THRESHOLD = 0.5
MAX_ANCHORS = 4


def find_anchor_tokens(input_ids, attention_mask, peak_abs_pos):
    """Embed all non-pad tokens in the 512-token input, find up to MAX_ANCHORS tokens
    most similar to the peak token (above threshold).

    Shards are left-padded: pad tokens (attention_mask=0) are at the start.

    Returns: (peak_token_str, anchors_list)
    """
    # Build list of (original_index, token_string) for non-pad tokens only
    real_tokens = []
    for i, (tid, mask) in enumerate(zip(input_ids, attention_mask)):
        if mask == 1:
            real_tokens.append((i, tokenizer.decode([tid])))

    # Find peak token's position in the real_tokens list
    peak_real_idx = None
    peak_token_str = tokenizer.decode([input_ids[peak_abs_pos]])
    for ri, (orig_idx, tok_str) in enumerate(real_tokens):
        if orig_idx == peak_abs_pos:
            peak_real_idx = ri
            break

    if peak_real_idx is None:
        # Peak landed on a pad token (shouldn't happen), return just the peak string
        return peak_token_str, []

    # Embed all real token strings in one batch
    token_strings = [ts for _, ts in real_tokens]
    embeddings = get_embeddings(token_strings)
    peak_embedding = embeddings[peak_real_idx]

    # Compute similarity of every real token to the peak token
    similarities = []
    for ri, (orig_idx, tok_str) in enumerate(real_tokens):
        if ri == peak_real_idx:
            continue
        sim = cosine_similarity(peak_embedding, embeddings[ri])
        similarities.append((orig_idx, sim, tok_str))

    # Sort by similarity descending, take top that clear threshold
    similarities.sort(key=lambda x: x[1], reverse=True)
    anchors = []
    seen_strings = {peak_token_str.strip().lower()}
    for orig_idx, sim, tok_str in similarities:
        if sim < ANCHOR_SIMILARITY_THRESHOLD:
            break
        normalized = tok_str.strip().lower()
        if normalized in seen_strings or not normalized:
            continue
        seen_strings.add(normalized)
        anchors.append({"token_str": tok_str, "similarity": sim, "position": orig_idx})
        if len(anchors) >= MAX_ANCHORS:
            break

    return peak_token_str, anchors


# --- Scoring helpers ---

def anchor_token_score(peak_token_str, anchors, prediction_text):
    """Score: what fraction of anchor tokens (including peak) appear in prediction?"""
    all_anchors = [peak_token_str] + [a["token_str"] for a in anchors]
    if not all_anchors:
        return 0.0
    found = sum(1 for a in all_anchors if a in prediction_text)
    return found / len(all_anchors)


def lcs_length(seq_a, seq_b):
    """Longest common subsequence length between two sequences."""
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def sequence_similarity(actual_text, prediction_text):
    """LCS ratio of tokenized sequences."""
    actual_ids = tokenizer.encode(actual_text, add_special_tokens=False)
    pred_ids = tokenizer.encode(prediction_text, add_special_tokens=False)
    if len(actual_ids) == 0:
        return 0.0
    lcs_len = lcs_length(actual_ids, pred_ids)
    return lcs_len / len(actual_ids)


def normalize_text(text):
    """Collapse whitespace and standardize quotes for fuzzy matching."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def fuzzy_find_position(plain_text, substring):
    """Find the best match position of substring in plain_text using fuzzy matching."""
    norm_plain = normalize_text(plain_text)
    norm_sub = normalize_text(substring)

    idx = norm_plain.find(norm_sub)
    if idx != -1:
        return idx

    matcher = difflib.SequenceMatcher(None, norm_plain, norm_sub, autojunk=False)
    match = matcher.find_longest_match(0, len(norm_plain), 0, len(norm_sub))
    if match.size > 0:
        return match.a
    return None


def distance_between(plain_text, pos_a, pos_b):
    """Absolute character distance between two positions."""
    if pos_a is None or pos_b is None:
        return len(plain_text)
    return abs(pos_a - pos_b)


def semantic_score(cosine_sim, distance, plain_text_len):
    """Distance-sensitized semantic similarity, rescaled to 0-1."""
    decay_scale = plain_text_len * 0.3
    if decay_scale == 0:
        distance_weight = 1.0
    else:
        distance_weight = math.exp(-distance / decay_scale)
    sensitized = cosine_sim * distance_weight
    score = max(0.0, (sensitized - 0.15) / 0.85)
    return min(1.0, score)


# --- Main ---

simulation_files = sorted(glob.glob("explanations/*_simulation.json"))
print(f"Found {len(simulation_files)} simulation files")

for filepath in simulation_files:
    with open(filepath) as f:
        data = json.load(f)

    neuron_id = data["neuron_id"]
    results = data["results"]
    scored_results = []

    for example in results:
        actual_text = example["actual_text"]
        plain_text = example["plain_text"]
        predictions = example["predictions"]
        shard_id = example["shard_id"]
        row_idx = example["row_idx"]

        # 1. Load 512-token input_ids + attention_mask from shard
        input_ids, attention_mask = load_shard_row(shard_id, row_idx)

        # 2. Find where the 33-token window sits in the 512-token input
        window_start = find_window_in_input_ids(input_ids, actual_text)
        if window_start is not None:
            peak_abs_pos = window_start + 16
        else:
            print(f"  WARNING: Could not locate window for {neuron_id} ex{example['example_idx']}")
            peak_abs_pos = 256

        # 3. Embed all non-pad tokens, find anchor tokens similar to peak
        peak_token_str, anchors = find_anchor_tokens(input_ids, attention_mask, peak_abs_pos)
        anchor_info = ", ".join(
            "{0}({1:.2f})".format(a["token_str"].strip(), a["similarity"]) for a in anchors
        )
        print(f"    ex{example['example_idx']}: peak=\"{peak_token_str.strip()}\", anchors=[{anchor_info}]")

        # 4. Batch embed actual_text + all prediction texts for semantic similarity
        texts_to_embed = [actual_text] + [p["text"] for p in predictions]
        text_embeddings = get_embeddings(texts_to_embed)
        actual_embedding = text_embeddings[0]

        # 5. Fuzzy-find actual_text position in plain_text
        actual_pos = fuzzy_find_position(plain_text, actual_text)

        scored_pairs = []
        for i, pred in enumerate(predictions):
            pred_text = pred["text"]
            pred_embedding = text_embeddings[i + 1]

            # Signal 1: Anchor token score
            ats = anchor_token_score(peak_token_str, anchors, pred_text)

            # Signal 2: Sequence similarity (LCS ratio)
            seq_sim = sequence_similarity(actual_text, pred_text)

            # Signal 3: Distance-sensitized semantic similarity
            pred_pos = fuzzy_find_position(plain_text, pred_text)
            dist = distance_between(plain_text, actual_pos, pred_pos)
            cos_sim = cosine_similarity(actual_embedding, pred_embedding)
            sem_score = semantic_score(cos_sim, dist, len(plain_text))

            # Combined: max of all three
            combined = max(ats, seq_sim, sem_score)

            scored_pairs.append({
                "prediction_rank": pred["rank"],
                "peak_token": peak_token_str.strip(),
                "anchor_tokens": [a["token_str"].strip() for a in anchors],
                "anchor_token_score": round(ats, 4),
                "sequence_similarity": round(seq_sim, 4),
                "cosine_similarity": round(cos_sim, 6),
                "distance_chars": dist,
                "semantic_score": round(sem_score, 4),
                "combined_score": round(combined, 4),
            })

        scored_results.append({
            "example_idx": example["example_idx"],
            "actual_text": actual_text,
            "normalized_activation": example["normalized"],
            "pairs": scored_pairs,
        })

    # Write output
    out_path = filepath.replace("_simulation.json", "_unified_scores.json")
    output = {
        "neuron_id": neuron_id,
        "model": data.get("model"),
        "explanation_used": data.get("explanation_used"),
        "results": scored_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  {neuron_id}: {len(scored_results)} examples -> {os.path.basename(out_path)}")

print("Done.")
