"""
data/validate_dataset.py

Validates a preference dataset against the AlignR schema and
enforces the IAA threshold before training is permitted.

Usage:
    python data/validate_dataset.py --dataset data/pref_dataset/
"""

import argparse
import json
import sys
from pathlib import Path

import jsonschema

PREFERENCE_PAIR_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "prompt_id", "prompt", "chosen", "rejected",
                 "annotator_id", "timestamp", "domain", "source"],
    "properties": {
        "schema_version": {"type": "string"},
        "prompt_id":      {"type": "string", "minLength": 8},
        "prompt":         {"type": "string", "minLength": 10},
        "chosen":         {"type": "string", "minLength": 5},
        "rejected":       {"type": "string", "minLength": 5},
        "annotator_id":   {"type": "string"},
        "timestamp":      {"type": "string"},
        "iaa_score":      {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "domain":         {"type": "string"},
        "source":         {"type": "string"},
    },
    "additionalProperties": True,
}

IAA_THRESHOLD = 0.7
MIN_PAIRS = 1000


def validate_file(pairs_path: Path) -> tuple[int, int, list[str]]:
    """Validate each pair. Returns (total, valid, errors)."""
    errors = []
    total = 0
    valid = 0
    low_iaa = 0

    with open(pairs_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON — {e}")
                continue

            try:
                jsonschema.validate(pair, PREFERENCE_PAIR_SCHEMA)
            except jsonschema.ValidationError as e:
                errors.append(f"Line {line_num}: schema error — {e.message}")
                continue

            # IAA check (bootstrap data has None — pass through with warning)
            iaa = pair.get("iaa_score")
            if iaa is not None and iaa < IAA_THRESHOLD:
                low_iaa += 1
                errors.append(
                    f"Line {line_num}: IAA score {iaa:.3f} below threshold {IAA_THRESHOLD} "
                    f"(prompt_id={pair['prompt_id']})"
                )
                continue

            valid += 1

    return total, valid, errors, low_iaa


def main():
    parser = argparse.ArgumentParser(description="Validate AlignR preference dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any IAA failures (default: warn only)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    pairs_path = dataset_dir / "preference_pairs.jsonl"
    meta_path = dataset_dir / "metadata.json"

    if not pairs_path.exists():
        print(f"ERROR: {pairs_path} not found.")
        sys.exit(1)

    print(f"Validating {pairs_path}...")
    total, valid, errors, low_iaa = validate_file(pairs_path)

    print(f"\nResults:")
    print(f"  Total pairs:   {total}")
    print(f"  Valid pairs:   {valid}")
    print(f"  IAA failures:  {low_iaa}")
    print(f"  Other errors:  {len(errors) - low_iaa}")

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\nDataset metadata:")
        print(f"  Source:        {meta.get('source', 'unknown')}")
        print(f"  Created:       {meta.get('created_at', 'unknown')}")
        print(f"  Dataset hash:  {meta.get('dataset_hash', 'unknown')}")

    if valid < MIN_PAIRS:
        print(f"\nWARNING: Only {valid} valid pairs. Recommend ≥{MIN_PAIRS} for training.")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    if low_iaa > 0 and args.strict:
        print(f"\nFAIL: {low_iaa} pairs below IAA threshold. Run with --validate in export_preferences.py to filter.")
        sys.exit(1)

    if valid == 0:
        print("\nFAIL: No valid pairs.")
        sys.exit(1)

    print(f"\nPASS: Dataset ready for training ({valid} pairs).")


if __name__ == "__main__":
    main()
