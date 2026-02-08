"""
Generate neuron explanations using an explainer model via OpenRouter.

Reads paired topk/bottomk parquet files from the activation pipeline,
samples high- and low-activation examples, and asks the explainer model
to describe what the neuron detects. Results are saved as JSON per neuron.
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

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_INPUT_DIR = Path("activation_outputs")
DEFAULT_OUTPUT_DIR = Path("explanations")

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
# Core logic
# ---------------------------------------------------------------------------

def _format_examples(rows: List[Dict[str, Any]]) -> str:
    """Format a list of sampled rows into prompt lines."""
    return "\n".join(
        f'Activation: {row["normalized"]:.2f} | Text: "{row["text"]}"'
        for row in rows
    )


def explain_neuron(
    neuron_id: str,
    input_dir: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
) -> Path:
    """
    Load parquets for *neuron_id*, sample examples, call the explainer model,
    and write the result as JSON.

    Returns the path to the output JSON file.
    """
    topk_path = input_dir / f"{neuron_id}_topk.parquet"
    bottomk_path = input_dir / f"{neuron_id}_bottomk.parquet"

    for p in (topk_path, bottomk_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing activation file: {p}")

    # Load tables
    topk_table = pq.read_table(topk_path)
    bottomk_table = pq.read_table(bottomk_path)

    rng = random.Random(42069)

    # Sample rows (or take all if fewer than requested)
    def _sample(table, n: int) -> List[Dict[str, Any]]:
        all_rows = []
        normalized = table.column("normalized").to_pylist()
        texts = table.column("text").to_pylist()
        for i in range(table.num_rows):
            all_rows.append({
                "normalized": float(normalized[i]),
                "text": clean_text(texts[i]),
            })
        if len(all_rows) <= n:
            return all_rows
        return rng.sample(all_rows, n)

    high_rows = _sample(topk_table, 30)
    low_rows = _sample(bottomk_table, 20)

    # Sort by activation descending / ascending for readability
    high_rows.sort(key=lambda r: r["normalized"], reverse=True)
    low_rows.sort(key=lambda r: r["normalized"])

    prompt = PROMPT_TEMPLATE.format(
        high_examples=_format_examples(high_rows),
        low_examples=_format_examples(low_rows),
    )

    # Call explainer model
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

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{neuron_id}.json"
    payload = {
        "neuron_id": neuron_id,
        "model": client.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "high_examples": high_rows,
        "low_examples": low_rows,
        "prompt": prompt,
        "explanation": explanation,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain neurons using high/low activation examples"
    )
    parser.add_argument(
        "--neuron",
        required=True,
        help="Neuron ID(s), e.g. '0_42' or '0_42,17_2410'",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    neuron_ids = [n.strip() for n in args.neuron.split(",") if n.strip()]

    if len(neuron_ids) == 1:
        nid = neuron_ids[0]
        print(f"[{nid}] explaining...")
        try:
            out = explain_neuron(nid, args.input_dir, args.output_dir, args.model)
            print(f"[{nid}] done -> {out}")
        except Exception as exc:
            print(f"[{nid}] ERROR: {exc}")
    else:
        workers = min(args.max_workers, len(neuron_ids))
        print(f"Explaining {len(neuron_ids)} neurons with {workers} threads")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(explain_neuron, nid, args.input_dir, args.output_dir, args.model): nid
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
