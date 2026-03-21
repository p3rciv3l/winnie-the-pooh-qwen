import math
import json
import glob
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def dot(e1, e2):
    return sum(x * y for x, y in zip(e1, e2))

def mag(e):
    return math.sqrt(sum(c ** 2 for c in e))

def cosine_similarity(e1, e2):
    return dot(e1, e2) / (mag(e1) * mag(e2))

endpoint = "https://lpr-dev.openai.azure.com/"
deployment = "text-embedding-3-large"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_KEY")
)

def get_embeddings(texts):
    response = client.embeddings.create(input=texts, model=deployment)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

def find_slice(plain_text, substring):
    start = plain_text.find(substring)
    if start == -1:
        return None
    return (start, start + len(substring))

def slices_overlap(slice_a, slice_b):
    if slice_a is None or slice_b is None:
        return False
    return slice_a[0] < slice_b[1] and slice_b[0] < slice_a[1]

if __name__ == "__main__":
    simulation_files = sorted(glob.glob("explanations/data/*_simulation.json"))
    print(f"Found {len(simulation_files)} simulation files")

    for filepath in simulation_files:
        with open(filepath) as f:
            data = json.load(f)

        neuron_id = data["neuron_id"]
        results = data["results"]
        similarity_results = []

        for example in results:
            actual_text = example["actual_text"]
            plain_text = example["plain_text"]
            predictions = example["predictions"]

            # Collect all texts to embed in one batch: actual_text + all prediction texts
            texts_to_embed = [actual_text] + [p["text"] for p in predictions]
            embeddings = get_embeddings(texts_to_embed)

            actual_embedding = embeddings[0]
            actual_slice = find_slice(plain_text, actual_text)

            pairs = []
            for i, pred in enumerate(predictions):
                pred_embedding = embeddings[i + 1]
                pred_slice = find_slice(plain_text, pred["text"])
                sim = cosine_similarity(actual_embedding, pred_embedding)
                overlap = slices_overlap(actual_slice, pred_slice)

                pairs.append({
                    "prediction_rank": pred["rank"],
                    "cosine_similarity": round(sim, 6),
                    "actual_text_slice": list(actual_slice) if actual_slice else None,
                    "prediction_text_slice": list(pred_slice) if pred_slice else None,
                    "slices_overlap": overlap,
                })

            similarity_results.append({
                "example_idx": example["example_idx"],
                "actual_text": actual_text,
                "normalized_activation": example["normalized"],
                "pairs": pairs,
            })

        # Write output
        out_path = filepath.replace("_simulation.json", "_similarity.json")
        output = {
            "neuron_id": neuron_id,
            "model": data.get("model"),
            "explanation_used": data.get("explanation_used"),
            "results": similarity_results,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"  {neuron_id}: {len(similarity_results)} examples -> {os.path.basename(out_path)}")

    print("Done.")
