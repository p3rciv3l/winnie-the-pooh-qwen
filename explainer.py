# TODO: Modify prompt to read from activation scores
# TODO: Modify prompt to include activation examples

"""
Generate neuron explanations using an explainer model via OpenRouter.

Reads per-neuron activation parquet files and sends the top activating sequences
to DeepSeek for explanation. Results are saved per neuron as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

from clients.openrouter_client import get_model_client

DEFAULT_MODEL = "deepseek-v3.2"
DEFAULT_TOP_N = 35
DEFAULT_INPUT_DIR = Path("activation_outputs")
DEFAULT_OUTPUT_DIR = Path("explanations")

PROMPT_TEMPLATE = """\
Below are text segments that caused this feature to activate most strongly.  Activation scores range from 0-10, where higher values indicate stronger  matches. Analyze these examples and explain what concept or pattern this feature detects.

=== ACTIVATION DATA ===
[INSERT 35 EXAMPLES HERE IN THE FOLLOWING FORMAT]
Activation: [ACTIVATION SCORE] | Text: “[INSERT TEXT HERE]”
Activation: [ACTIVATION SCORE] | Text: “[INSERT TEXT HERE]”
Activation: [ACTIVATION SCORE] | Text: “[INSERT TEXT HERE]”
...
=== END DATA ===

Structure your explanation as follows:

1. **Opening statement**: Name the core pattern and provide clarifying examples in parentheses
   Example: "This neuron strongly responds to [pattern category] (such as [example type 1], [example type 2], or [example type 3])..."

2. **Domain coverage**: Be aware that this neuron may detect:
- Political figures, parties, or movements
- Ideological terminology (e.g., socialism, democracy, nationalism)
- Political events or historical references
- Governance concepts or policy discussions
- Sensitive political topics across different countries
among other domains. You should note this.
Generally you should try to be specific when discussing these domains, if the examples you read allow for this.

4. **Closing interpretation**: Summarize the underlying linguistic and semantic focus
   Example: "This indicates a focus on [pattern] across [domains]."

Guidelines:
- Include examples from multiple languages/formats when the data supports it
- Note activation strength when relevant ("strongly responds to", "also processes")
- Keep explanations to 3-6 sentences unless categorization requires more detail

The activation format has values 0-10. Higher values = stronger matches.
"""

# Thread-local session for connection pooling
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


def normalize_activations(activations: Sequence[float]) -> List[float]:
    if not activations:
        return []
    min_val = min(activations)
    max_val = max(activations)
    if max_val == min_val:
        return [5.0] * len(activations)
    return [((a - min_val) / (max_val - min_val)) * 10.0 for a in activations]


def load_examples(file_path: Path, top_n: int) -> List[Dict[str, Any]]:
    table = pq.read_table(file_path)
    columns = set(table.column_names)

    activations = table.column("activation").to_pylist()
    texts = table.column("text").to_pylist()
    ranks = table.column("rank").to_pylist() if "rank" in columns else None
    normalized = table.column("normalized").to_pylist() if "normalized" in columns else None

    if normalized is None:
        normalized = normalize_activations(activations)

    if ranks is not None:
        order = sorted(range(len(texts)), key=lambda i: ranks[i])
    else:
        order = sorted(range(len(texts)), key=lambda i: activations[i], reverse=True)

    examples = []
    for idx in order[:top_n]:
        examples.append({
            "activation": float(activations[idx]),
            "normalized": float(normalized[idx]),
            "text": clean_text(texts[idx]),
        })
    return examples


def build_prompt(examples: Sequence[Dict[str, Any]]) -> str:
    lines = [
        f'Activation: {ex["normalized"]:.2f} | Text: "{ex["text"]}"'
        for ex in examples
    ]
    return PROMPT_TEMPLATE.format(examples="\n".join(lines))


def write_output(
    output_path: Path,
    neuron_id: str,
    model: str,
    examples: Sequence[Dict[str, Any]],
    explanation: str,
    prompt: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "neuron_id": neuron_id,
        "model": model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "examples": list(examples),
        "prompt": prompt,
        "explanation": explanation,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def extract_response_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content") or message.get("reasoning") or ""


def explain_neuron(
    file_path: Path,
    client,
    top_n: int,
    output_dir: Path,
    overwrite: bool,
) -> Tuple[str, str]:
    neuron_id = file_path.stem
    output_path = output_dir / f"{neuron_id}.json"

    if output_path.exists() and not overwrite:
        return neuron_id, "skipped"

    examples = load_examples(file_path, top_n)
    if not examples:
        return neuron_id, "empty"

    prompt = build_prompt(examples)
    response = client.generate(
        [{"role": "user", "content": prompt}],
        session=get_session(),
    )
    explanation = extract_response_text(response)
    if not explanation:
        return neuron_id, "no_response"

    write_output(
        output_path=output_path,
        neuron_id=neuron_id,
        model=client.model,
        examples=examples,
        explanation=explanation,
        prompt=prompt,
    )
    return neuron_id, "ok"


def load_neuron_filter(
    neuron_list_path: Optional[Path],
    neuron_csv: Optional[str],
) -> Optional[set[str]]:
    neurons: set[str] = set()
    invalid: set[str] = set()
    if neuron_list_path:
        for line in neuron_list_path.read_text().splitlines():
            line = line.strip()
            if line:
                if is_valid_neuron_id(line):
                    neurons.add(line)
                else:
                    invalid.add(line)
    if neuron_csv:
        for item in neuron_csv.split(","):
            item = item.strip()
            if item:
                if is_valid_neuron_id(item):
                    neurons.add(item)
                else:
                    invalid.add(item)
    if invalid:
        raise ValueError(
            "Invalid neuron id(s); expected format '<layer>_<neuron>': "
            + ", ".join(sorted(invalid))
        )
    return neurons or None


def is_valid_neuron_id(neuron_id: str) -> bool:
    return re.fullmatch(r"\d+_\d+", neuron_id) is not None


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    try:
        data = json.loads(checkpoint_path.read_text())
        return set(data.get("completed", []))
    except Exception:
        return set()


def save_checkpoint(checkpoint_path: Path, completed: set[str]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps({"completed": sorted(completed)}, indent=2, ensure_ascii=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate neuron explanations")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_DIR / "checkpoint.json")
    parser.add_argument("--neuron-list", type=Path, default=None)
    parser.add_argument("--neurons", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    neuron_filter = load_neuron_filter(args.neuron_list, args.neurons)
    completed = load_checkpoint(args.checkpoint)

    files = sorted(args.input_dir.glob("*.parquet"))
    if neuron_filter:
        files = [f for f in files if f.stem in neuron_filter]

    if not files:
        print("No activation files found.")
        return

    client = get_model_client(args.model)
    lock = threading.Lock()

    def worker(file_path: Path) -> Tuple[str, str]:
        neuron_id = file_path.stem
        output_path = args.output_dir / f"{neuron_id}.json"
        if neuron_id in completed and not args.overwrite:
            return neuron_id, "skipped"
        if output_path.exists() and not args.overwrite:
            return neuron_id, "skipped"
        return explain_neuron(
            file_path=file_path,
            client=client,
            top_n=args.top_n,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )

    total = len(files)
    results = {"ok": 0, "skipped": 0, "empty": 0, "no_response": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(worker, f): f for f in files}
        for future in tqdm(future_map, total=total, desc="Explaining neurons"):
            file_path = future_map[future]
            neuron_id = file_path.stem
            try:
                neuron_id, status = future.result()
            except Exception as exc:
                status = "error"
                print(f"[{neuron_id}] error: {exc}")
            results[status] = results.get(status, 0) + 1
            if status == "ok":
                with lock:
                    completed.add(neuron_id)
                    save_checkpoint(args.checkpoint, completed)

    print("Done.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()