"""
Run simulation evaluation for neuron explanations.

Scans the explanations directory for completed explanation files ({neuron_id}.json),
skips any that already have a corresponding _simulation.json, and runs simulation
for the rest. Re-derives holdout rows using the same sampling seed as explainer.py.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from clients.openrouter_client import get_model_client

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_INPUT_DIR = Path("activation_outputs")
DEFAULT_OUTPUT_DIR = Path("explanations/data")
DEFAULT_SHARDS_DIR = Path("shards")

SIMULATION_PROMPT_TEMPLATE = """\
You are evaluating a neuron explanation by predicting activation patterns.

A neuron in a sparse autoencoder trained on a large language model has been described as:

=== NEURON EXPLANATION ===
{explanation}
=== END EXPLANATION ===

Below are {n} text examples that were shown to this neuron. For each example, identify \
3 non-overlapping contiguous sections of approximately 33 tokens (~2-3 sentences) that you \
believe caused the neuron to activate most strongly, ranked by confidence. \
Copy the text verbatim from the example.

{examples}

Respond with ONLY a JSON array. Each entry should contain:
- "example": the example number (1-indexed)
- "predictions": an array of exactly 3 objects, each with:
  - "rank": 1, 2, or 3 (1 = most confident)
  - "text": the exact verbatim ~33-token section from the example
  - "concept": a single word or short phrase that represents what you think was the highest activating concept in this example
  - "reasoning": one sentence explaining why this section matches the neuron's pattern

Example prediction object:
{{"rank": 1, "text": "...", "concept": "geopolitical", "reasoning": "..."}}

Respond with only the JSON array, no other text.
"""

_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
            ),
        )
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def extract_response_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content") or message.get("reasoning") or ""


def _sample_rows(table, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    all_rows = []
    normalized = table.column("normalized").to_pylist()
    texts = table.column("text").to_pylist()
    shard_ids = table.column("shard_id").to_pylist()
    row_idxs = table.column("row_idx").to_pylist()

    for i in range(table.num_rows):
        all_rows.append({
            "normalized": float(normalized[i]),
            "text": clean_text(texts[i]),
            "shard_id": shard_ids[i],
            "row_idx": int(row_idxs[i]),
        })
    if len(all_rows) <= n:
        return all_rows
    return rng.sample(all_rows, n)


def _load_plain_text(shard_id: str, row_idx: int, shards_dir: Path) -> str:
    shard_path = shards_dir / f"{shard_id}.parquet"
    table = pq.read_table(shard_path, columns=["plain_text"])
    return str(table.column("plain_text")[row_idx].as_py())


def _derive_holdout_rows(neuron_id: str, input_dir: Path) -> List[Dict[str, Any]]:
    """Re-derive holdout rows using the same sampling logic and seed as explainer.py."""
    topk_path = input_dir / f"{neuron_id}_topk.parquet"
    bottomk_path = input_dir / f"{neuron_id}_bottomk.parquet"

    for p in (topk_path, bottomk_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing activation file: {p}")

    topk_table = pq.read_table(topk_path)
    bottomk_table = pq.read_table(bottomk_path)

    rng = random.Random(42069)
    high_rows = _sample_rows(topk_table, 33, rng)
    low_rows = _sample_rows(bottomk_table, 22, rng)

    rng.shuffle(high_rows)
    rng.shuffle(low_rows)

    holdout_high = high_rows[:3]
    holdout_low = low_rows[:2]
    return holdout_high + holdout_low


def simulate_neuron(
    neuron_id: str,
    explanation: str,
    holdout_rows: List[Dict[str, Any]],
    shards_dir: Path,
    output_dir: Path,
    model: str,
) -> Path:
    plain_texts = []
    for row in holdout_rows:
        text = _load_plain_text(row["shard_id"], row["row_idx"], shards_dir)
        plain_texts.append(text)

    examples_block = "\n\n".join(
        f"=== Example {i + 1} ===\n{text}\n=== End Example {i + 1} ==="
        for i, text in enumerate(plain_texts)
    )

    prompt = SIMULATION_PROMPT_TEMPLATE.format(
        explanation=explanation,
        n=len(holdout_rows),
        examples=examples_block,
    )

    client = get_model_client(model)
    response = client.generate(
        [{"role": "user", "content": prompt}],
        session=get_session(),
    )
    prediction_text = extract_response_text(response)

    predictions = []
    try:
        cleaned = prediction_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        predictions = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        predictions = [{"raw_response": prediction_text}]

    results = []
    for i, row in enumerate(holdout_rows):
        pred = predictions[i] if i < len(predictions) else {}
        results.append({
            "example_idx": i + 1,
            "shard_id": row["shard_id"],
            "row_idx": row["row_idx"],
            "normalized": row["normalized"],
            "actual_text": row["text"],
            "plain_text": plain_texts[i],
            "predictions": pred.get("predictions", []),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    sim_path = output_dir / f"{neuron_id}_simulation.json"
    payload = {
        "neuron_id": neuron_id,
        "model": client.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "explanation_used": explanation,
        "results": results,
    }
    sim_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return sim_path


def simulate_neuron_from_file(
    neuron_id: str,
    input_dir: Path,
    output_dir: Path,
    shards_dir: Path,
    model: str,
) -> Path:
    explanation_path = output_dir / f"{neuron_id}.json"
    with open(explanation_path) as f:
        data = json.load(f)
    explanation = data["explanation"]
    holdout_rows = _derive_holdout_rows(neuron_id, input_dir)
    return simulate_neuron(
        neuron_id=neuron_id,
        explanation=explanation,
        holdout_rows=holdout_rows,
        shards_dir=shards_dir,
        output_dir=output_dir,
        model=model,
    )


def find_explained_neurons(output_dir: Path) -> List[str]:
    """Return all neuron IDs that have an explanation file (simulation files will be overwritten)."""
    return sorted(
        p.stem
        for p in output_dir.glob("*.json")
        if not p.stem.endswith("_simulation") and not p.stem.endswith("_unified_scores")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simulation for all explanation files that don't yet have a simulation"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument(
        "--neurons",
        nargs="+",
        metavar="NEURON_ID",
        help="Neuron IDs to simulate. Defaults to the hardcoded list.",
    )
    args = parser.parse_args()

    if args.neurons:
        neuron_ids = args.neurons
    else:
        empty_predictions_file = Path(__file__).parent / "empty_predictions.txt"
        neuron_ids = empty_predictions_file.read_text().splitlines()

    if not neuron_ids:
        print("No neurons to simulate.")
        return

    print(f"Found {len(neuron_ids)} neurons to simulate.")

    workers = min(args.max_workers, len(neuron_ids))
    print(f"Using {workers} threads")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                simulate_neuron_from_file, nid, args.input_dir, args.output_dir, args.shards_dir, args.model
            ): nid
            for nid in neuron_ids
        }
        with tqdm(total=len(futures), unit="neuron") as pbar:
            for future in futures:
                nid = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    tqdm.write(f"[{nid}] ERROR: {exc}")
                pbar.update(1)


if __name__ == "__main__":
    main()
