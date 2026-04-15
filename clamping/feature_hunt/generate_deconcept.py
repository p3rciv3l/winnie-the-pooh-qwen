"""
Generate de-concept prompts using the Concept Contrastive Query Pairs method.

For each China-sensitive test prompt, generates a matched prompt that preserves
the linguistic structure but removes China-specific content. This follows the
approach from the Safe-SAIL paper (Section 2.2.1 / Appendix B.1).

Usage:
    export OPENROUTER_API_KEY=...
    python generate_deconcept.py \
        --input ../example_questions/prompts.txt \
        --output deconcept_prompts.txt \
        --n 200
"""

import argparse
import json
import random
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from clients.openrouter_client import OpenRouterClient

SYSTEM_PROMPT = """\
You will be provided with a query about a China-sensitive political topic \
(e.g. Tiananmen Square, Taiwan, Uyghurs, Tibet, Hong Kong protests, CCP, \
Chinese dissidents, censorship in China).

Your task: generate a new query that:
1. Preserves the EXACT same grammatical structure, length, and question type
2. Replaces all China-specific references with equivalent references to \
   OTHER countries, events, or people (e.g. Tiananmen → Kent State, \
   Taiwan → Kosovo, Uyghurs → Kurds, Xi Jinping → Erdogan)
3. Keeps the same level of political sensitivity — the replacement topic \
   should also involve human rights, political repression, sovereignty \
   disputes, or historical atrocities, just NOT about China
4. Does NOT water down the question — if the original asks about a \
   "massacre", the replacement should also ask about a "massacre"

Output ONLY the rewritten query, nothing else. No quotes, no explanation."""


def generate_deconcept(client, prompt):
    """Generate a single de-concept prompt."""
    response = client.generate(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=512,
        logprobs=False,
        reasoning=None,
    )
    choices = response.get("choices", [])
    if not choices:
        return None
    return choices[0]["message"]["content"].strip().strip('"').strip("'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to test prompts")
    parser.add_argument("--output", required=True, help="Output path for de-concept prompts")
    parser.add_argument("--n", type=int, default=200, help="Number of prompts to process")
    parser.add_argument("--model", default="deepseek/deepseek-chat", help="OpenRouter model")
    parser.add_argument("--seed", type=int, default=42069, help="Random seed for selection")
    args = parser.parse_args()

    with open(args.input) as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    rng = random.Random(args.seed)
    if args.n < len(all_prompts):
        selected = rng.sample(all_prompts, args.n)
    else:
        selected = all_prompts

    print(f"Generating de-concept versions for {len(selected)} prompts using {args.model}")

    client = OpenRouterClient(model=args.model, seed=args.seed)

    results = []
    for i, prompt in enumerate(selected):
        deconcept = generate_deconcept(client, prompt)
        if deconcept is None:
            print(f"  [{i+1}/{len(selected)}] FAILED: {prompt[:60]}...")
            continue
        results.append({"concept": prompt, "deconcept": deconcept})
        print(f"  [{i+1}/{len(selected)}] {prompt[:50]}...")
        print(f"    → {deconcept[:50]}...")

    # Save as JSONL (paired format) and also flat txt files
    jsonl_path = args.output.replace(".txt", ".jsonl")
    concept_path = args.output.replace(".txt", "_concept.txt")
    deconcept_path = args.output.replace(".txt", "_deconcept.txt")

    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(concept_path, "w") as f:
        for r in results:
            f.write(r["concept"] + "\n")

    with open(deconcept_path, "w") as f:
        for r in results:
            f.write(r["deconcept"] + "\n")

    print(f"\nDone. {len(results)} pairs saved:")
    print(f"  Paired:     {jsonl_path}")
    print(f"  Concept:    {concept_path}")
    print(f"  De-concept: {deconcept_path}")


if __name__ == "__main__":
    main()
