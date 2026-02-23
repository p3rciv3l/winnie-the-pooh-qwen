"""
Generate neuron explanations and run simulation evaluation via OpenRouter.

1. Explanation: Sample high/low activation examples, ask the explainer model
   to describe what the neuron detects.
2. Simulation: Hold out examples, give the model the explanation + original texts,
   ask it to predict the highest-activating 2-3 sentences, compare to ground truth.
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

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from clients.openrouter_client import get_model_client

from activation_collector.config import NEURONS

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_INPUT_DIR = Path("activation_outputs")
DEFAULT_OUTPUT_DIR = Path("explanations")
DEFAULT_SHARDS_DIR = Path("shards")


PROMPT_TEMPLATE = """\
You are analyzing a neuron from a sparse autoencoder trained on a large language model. \
Below are text segments split into two groups: those that caused the neuron to activate \
strongly (HIGH ACTIVATION) and those where the neuron barely fired (LOW ACTIVATION). \
Activation scores are normalized 0-10.

=== HIGH ACTIVATION EXAMPLES ===
{high_examples}
=== END HIGH ACTIVATION ===

=== LOW ACTIVATION EXAMPLES ===
{low_examples}
=== END LOW ACTIVATION ===

By contrasting what the neuron responds to strongly vs. weakly, explain what concept \
or pattern this neuron detects.

Structure your explanation as follows:

1. **Opening statement**: Name the core pattern and provide clarifying examples in parentheses
   Example: "This neuron strongly responds to [pattern category] (such as [example type 1], [example type 2], or [example type 3])..."

2. **Domain coverage**: Describe the domains and contexts where this pattern appears. \
Be specific — mention languages, topics, or text genres when the examples support it.

3. **Contrast with low-activation examples**: Explain what distinguishes the high-activation \
texts from the low-activation ones. What is present in the high group but absent in the low group?

4. **Closing interpretation**: Summarize the underlying linguistic or semantic focus in 1-2 sentences.

Guidelines:
- Include examples from multiple languages/formats when the data supports it
- Note activation strength when relevant ("strongly responds to", "also processes")
- Keep explanations to 3-6 sentences unless categorization requires more detail
"""

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

# ---------------------------------------------------------------------------
# Thread-local session for connection pooling
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def get_session() -> requests.Session:
    """Get or create a thread-local requests.Session with connection pooling."""
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


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_rows(table, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Sample n rows from a parquet table, keeping all fields for holdout use."""
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


def _format_examples(rows: List[Dict[str, Any]]) -> str:
    """Format a list of sampled rows into prompt lines."""
    return "\n".join(
        f'Activation: {row["normalized"]:.2f} | Text: "{row["text"]}"'
        for row in rows
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _load_plain_text(shard_id: str, row_idx: int, shards_dir: Path) -> str:
    """Load the original plain_text for a specific shard + row."""
    shard_path = shards_dir / f"{shard_id}.parquet"
    table = pq.read_table(shard_path, columns=["plain_text"])
    return str(table.column("plain_text")[row_idx].as_py())


def simulate_neuron(
    neuron_id: str,
    explanation: str,
    holdout_rows: List[Dict[str, Any]],
    shards_dir: Path,
    output_dir: Path,
    model: str,
) -> Path:
    """
    Run the simulation step and write results to a separate file.

    For each holdout example, loads the original plain_text from the shard,
    asks the model to predict 3 highest-activating ~33-token sections,
    and pairs predictions with the actual text from activation outputs.
    """
    # Load original plain_text for each holdout
    plain_texts = []
    for row in holdout_rows:
        text = _load_plain_text(row["shard_id"], row["row_idx"], shards_dir)
        plain_texts.append(text)

    # Build the examples block
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

    # Try to parse JSON predictions
    predictions = []
    try:
        cleaned = prediction_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        predictions = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        predictions = [{"raw_response": prediction_text}]

    # Build results pairing predictions with holdout metadata
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

    # Write simulation to separate file
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


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def explain_neuron(
    neuron_id: str,
    input_dir: Path,
    output_dir: Path,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    model: str = DEFAULT_MODEL,
    explain_only: bool = False,
) -> Path:
    """
    Explain a neuron and optionally run simulation evaluation.

    1. Sample 33 high / 22 low activation examples
    2. Hold out 3 high + 2 low for simulation
    3. Run explainer with remaining 30 high + 20 low
    4. (Unless explain_only) Run simulation: model predicts highest-activating sentences for holdouts
    5. Save explanation + simulation results as JSON
    """
    topk_path = input_dir / f"{neuron_id}_topk.parquet"
    bottomk_path = input_dir / f"{neuron_id}_bottomk.parquet"

    for p in (topk_path, bottomk_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing activation file: {p}")

    topk_table = pq.read_table(topk_path)
    bottomk_table = pq.read_table(bottomk_path)

    rng = random.Random(42069)

    # Sample 33 high, 22 low (with full row data)
    high_rows = _sample_rows(topk_table, 33, rng)
    low_rows = _sample_rows(bottomk_table, 22, rng)

    # Shuffle before splitting so holdouts are random
    rng.shuffle(high_rows)
    rng.shuffle(low_rows)

    # Split: holdout 3 high + 2 low, explain with the rest
    holdout_high = high_rows[:3]
    explain_high = high_rows[3:]
    holdout_low = low_rows[:2]
    explain_low = low_rows[2:]

    # Sort explanation examples for readability
    explain_high.sort(key=lambda r: r["normalized"], reverse=True)
    explain_low.sort(key=lambda r: r["normalized"])

    prompt = PROMPT_TEMPLATE.format(
        high_examples=_format_examples(explain_high),
        low_examples=_format_examples(explain_low),
    )

    # Step 1: Explanation
    client = get_model_client(model)
    response = client.generate(
        [{"role": "user", "content": prompt}],
        session=get_session(),
    )
    explanation = extract_response_text(response)
    if not explanation:
        raise RuntimeError(
            f"Empty response from model for neuron {neuron_id}. "
            f"Raw response: {json.dumps(response, indent=2)[:500]}"
        )

    # Write explanation output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{neuron_id}.json"
    payload = {
        "neuron_id": neuron_id,
        "model": client.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "high_examples": [
            {"normalized": r["normalized"], "text": r["text"]}
            for r in explain_high
        ],
        "low_examples": [
            {"normalized": r["normalized"], "text": r["text"]}
            for r in explain_low
        ],
        "prompt": prompt,
        "explanation": explanation,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Step 2: Simulation (separate file)
    if not explain_only:
        holdout_rows = holdout_high + holdout_low
        sim_path = simulate_neuron(
            neuron_id=neuron_id,
            explanation=explanation,
            holdout_rows=holdout_rows,
            shards_dir=shards_dir,
            output_dir=output_dir,
            model=model,
        )
        print(f"[{neuron_id}] simulation -> {sim_path}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain neurons and run simulation evaluation"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument(
        "--explain-only",
        action="store_true",
        help="Skip the simulation step; only generate and save explanations.",
    )
    parser.add_argument(
        "--neurons",
        nargs="+",
        metavar="NEURON_ID",
        help="Neuron IDs to process (e.g. 0_2516 0_268). Defaults to all neurons in config.",
    )
    args = parser.parse_args()

    neuron_ids = args.neurons if args.neurons else NEURONS

    mode = "explaining" if args.explain_only else "explaining + simulating"

    if len(neuron_ids) == 1:
        nid = neuron_ids[0]
        print(f"[{nid}] {mode}...")
        try:
            out = explain_neuron(nid, args.input_dir, args.output_dir, args.shards_dir, args.model, args.explain_only)
            print(f"[{nid}] done -> {out}")
        except Exception as exc:
            print(f"[{nid}] ERROR: {exc}")
    else:
        workers = min(args.max_workers, len(neuron_ids))
        print(f"{mode.capitalize()} {len(neuron_ids)} neurons with {workers} threads")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(explain_neuron, nid, args.input_dir, args.output_dir, args.shards_dir, args.model, args.explain_only): nid
                for nid in neuron_ids
            }
            for future in futures:
                nid = futures[future]
                try:
                    out = future.result()
                    print(f"[{nid}] done -> {out}")
                except Exception as exc:
                    print(f"[{nid}] ERROR: {exc}")


if __name__ == "__main__":
    main()
