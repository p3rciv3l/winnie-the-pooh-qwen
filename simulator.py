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
DEFAULT_OUTPUT_DIR = Path("explanations")
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
      neuron_ids = [
        "0_2516",
        "0_268",
        "0_2851",
        "8_14782",
        "8_15063",
        "8_15083",
        "8_15608",
        "8_16641",
        "8_17138",
        "8_17244",
        "8_17491",
        "8_17507",
        "8_17618",
        "8_17949",
        "8_1798",
        "8_18033",
        "8_18111",
        "8_18148",
        "8_18157",
        "8_18378",
        "8_18558",
        "8_18588",
        "8_18645",
        "8_18759",
        "8_18772",
        "8_18797",
        "8_18908",
        "8_18912",
        "8_19071",
        "8_19083",
        "8_19095",
        "8_19161",
        "8_19209",
        "8_19488",
        "8_19508",
        "8_19586",
        "8_19649",
        "8_19705",
        "8_1972",
        "8_19896",
        "8_20033",
        "8_20082",
        "8_20090",
        "8_201",
        "8_20131",
        "8_20145",
        "8_20239",
        "8_20253",
        "8_20269",
        "8_20308",
        "8_20319",
        "8_20420",
        "8_2079",
        "8_2128",
        "8_2213",
        "8_2303",
        "8_2331",
        "8_2371",
        "8_2398",
        "8_2547",
        "8_2579",
        "8_2618",
        "8_27",
        "8_2737",
        "8_2774",
        "8_2786",
        "8_2826",
        "8_2858",
        "8_2949",
        "8_2956",
        "8_3034",
        "8_3090",
        "8_3114",
        "8_318",
        "8_3214",
        "8_3296",
        "8_3308",
        "8_3415",
        "8_3505",
        "8_3546",
        "8_3623",
        "8_3645",
        "8_3658",
        "8_37",
        "8_3778",
        "8_3860",
        "8_3928",
        "8_393",
        "8_3934",
        "8_4095",
        "8_4247",
        "8_4266",
        "8_429",
        "8_4347",
        "8_4419",
        "8_4579",
        "8_4582",
        "8_4649",
        "8_4672",
        "8_4720",
        "8_4830",
        "8_4841",
        "8_4934",
        "8_5040",
        "8_5057",
        "8_5092",
        "8_5102",
        "8_5117",
        "8_5177",
        "8_5208",
        "8_5224",
        "8_5234",
        "8_5282",
        "8_5495",
        "8_5614",
        "8_5663",
        "8_5692",
        "8_5698",
        "8_5726",
        "8_5747",
        "8_589",
        "8_5923",
        "8_5994",
        "8_6146",
        "8_6171",
        "8_6213",
        "8_6249",
        "8_6259",
        "8_6292",
        "8_635",
        "8_6356",
        "8_6452",
        "8_6453",
        "8_6509",
        "8_6511",
        "8_6544",
        "8_659",
        "8_6596",
        "8_6640",
        "8_670",
        "8_6770",
        "8_6793",
        "8_690",
        "8_6987",
        "8_7037",
        "8_7040",
        "8_7056",
        "8_7079",
        "8_7099",
        "8_7172",
        "8_72",
        "8_7276",
        "8_7288",
        "8_790",
        "8_8026",
        "8_8196",
        "8_8224",
        "8_8259",
        "8_8272",
        "8_8288",
        "8_8322",
        "8_8333",
        "8_8373",
        "8_8377",
        "8_8652",
        "8_8701",
        "8_872",
        "8_8736",
        "8_8795",
        "8_8915",
        "8_8956",
        "8_9056",
        "8_9090",
        "8_9097",
        "8_911",
        "8_9149",
        "8_9166",
        "8_9227",
        "8_9233",
        "8_9270",
        "8_9342",
        "8_9405",
        "8_9467",
        "8_9554",
        "8_9555",
        "8_9585",
        "8_9592",
        "8_9600",
        "8_9664",
        "8_972",
        "8_9748",
        "8_9870",
        "8_989",
        "8_9907",
        "8_9983",
      ]

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
