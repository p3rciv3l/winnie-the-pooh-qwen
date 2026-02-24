"""
Retroactively mark each example in *_unified_scores.json files as
'high' or 'low' activation.

The holdout set is always 3 high (topk) + 2 low (bottomk), so:
- Sort the 5 examples by normalized_activation descending
- Top 3 → "high", bottom 2 → "low"
"""

from pathlib import Path
import json

EXPLANATIONS_DIR = Path("explanations")


def mark_file(path: Path) -> int:
    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return 0

    sorted_by_activation = sorted(
        results, key=lambda r: r["normalized_activation"], reverse=True
    )

    high_ids = {id(r) for r in sorted_by_activation[:3]}

    changed = 0
    for result in results:
        activation_type = "high" if id(result) in high_ids else "low"
        if result.get("activation_type") != activation_type:
            result["activation_type"] = activation_type
            changed += 1

    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return changed


def main() -> None:
    files = sorted(EXPLANATIONS_DIR.glob("*_unified_scores.json"))
    if not files:
        print("No *_unified_scores.json files found.")
        return

    total_files = 0
    total_examples = 0
    for path in files:
        changed = mark_file(path)
        if changed:
            total_files += 1
            total_examples += changed
            print(f"  {path.name}: marked {changed} examples")

    print(f"\nDone. Updated {total_examples} examples across {total_files} files.")


if __name__ == "__main__":
    main()
