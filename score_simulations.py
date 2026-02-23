import argparse
import json
import glob
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from similarity import get_embeddings, cosine_similarity, find_slice
from simulator import clean_text

parser = argparse.ArgumentParser()
parser.add_argument("--neurons", nargs="+", metavar="NEURON_ID", help="Specific neuron IDs to score.")
args = parser.parse_args()

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if args.neurons:
    simulation_files = sorted(f"explanations/{nid}_simulation.json" for nid in args.neurons)
else:
    simulation_files = sorted(glob.glob("explanations/*_simulation.json"))
print(f"Found {len(simulation_files)} simulation files")

for filepath in tqdm(simulation_files, unit="neuron"):
    with open(filepath) as f:
        data = json.load(f)

    neuron_id = data["neuron_id"]
    try:
        results = data["results"]
        scored_results = []

        for example in results:
            actual_text = example["actual_text"]
            plain_text = example["plain_text"]
            predictions = example["predictions"]

            # Peak token: tokenize actual_text, take middle token (index 16 of 33)
            token_ids = tokenizer.encode(actual_text, add_special_tokens=False)
            mid = len(token_ids) // 2
            peak_token = tokenizer.decode([token_ids[mid]])

            # Batch embed: [peak_token, actual_text, concept_1, pred_text_1, concept_2, pred_text_2, ...]
            texts_to_embed = [peak_token, actual_text]
            for pred in predictions:
                texts_to_embed.append(pred.get("concept", ""))
                texts_to_embed.append(pred["text"])
            embeddings = get_embeddings(texts_to_embed)

            peak_emb = embeddings[0]
            actual_emb = embeddings[1]

            # Find actual text slice in the source document
            actual_slice = find_slice(plain_text, actual_text)

            scored_pairs = []
            for i, pred in enumerate(predictions):
                concept_emb = embeddings[2 + i * 2]
                pred_text_emb = embeddings[2 + i * 2 + 1]

                # Signal 1: Concept score
                concept_score = cosine_similarity(concept_emb, peak_emb)

                # Signal 2: Meaning score
                meaning_score = cosine_similarity(actual_emb, pred_text_emb)

                # Signal 3: Overlap fraction — what fraction of actual_text is covered by pred
                # Normalize pred text to match clean_text applied to actual_text
                pred_text_clean = clean_text(pred.get("text", ""))
                actual_len = len(actual_text)
                if actual_text in pred_text_clean:
                    overlap_frac = 1.0
                elif pred_text_clean and pred_text_clean in actual_text:
                    overlap_frac = len(pred_text_clean) / actual_len if actual_len > 0 else 0.0
                else:
                    # Fallback: position-based overlap in plain_text
                    pred_slice = find_slice(plain_text, pred.get("text", ""))
                    if actual_slice and pred_slice:
                        intersection = max(0, min(actual_slice[1], pred_slice[1]) - max(actual_slice[0], pred_slice[0]))
                        overlap_frac = (intersection / actual_len) if actual_len > 0 else 0.0
                    else:
                        overlap_frac = 0.0

                location_score = overlap_frac * 0.425
                combined = concept_score * 0.15 + meaning_score * 0.425 + location_score

                scored_pairs.append({
                    "prediction_rank": pred["rank"],
                    "concept_score": round(concept_score, 4),
                    "meaning_score": round(meaning_score, 4),
                    "overlap_fraction": round(overlap_frac, 4),
                    "location_score": round(location_score, 4),
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
            json.dump(output, f, indent=2, ensure_ascii=False)

        tqdm.write(f"  {neuron_id}: {len(scored_results)} examples -> {os.path.basename(out_path)}")

    except Exception as e:
        tqdm.write(f"  ERROR [{neuron_id}]: {e}")

print("Done.")
