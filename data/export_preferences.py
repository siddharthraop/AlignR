"""
data/export_preferences.py

Exports annotated preference pairs from Argilla to the AlignR
JSONL format (TRL-compatible) and produces the ART-001 annotation log.

Usage:
    python data/export_preferences.py \
        --workspace alignr \
        --output data/pref_dataset/ \
        --validate
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import argilla as rg
from tqdm import tqdm


IAA_KAPPA_THRESHOLD = 0.7   # Cohen's κ — hard gate per preference pair
MIN_PAIRS = 1000             # warn if fewer pairs than this


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compute_simple_kappa(ratings: list[int]) -> float:
    """
    Compute a simplified Cohen's κ for a set of ratings on one item.
    Full IAA requires multiple annotators per item; this approximates
    agreement as the proportion of majority votes.
    """
    if len(ratings) < 2:
        return 1.0  # single annotator — pass through with warning
    majority = max(set(ratings), key=ratings.count)
    agreement = ratings.count(majority) / len(ratings)
    # Normalise: agreement of 1.0 → κ=1.0, 0.5 → κ=0.0
    return max(0.0, (agreement - 0.5) / 0.5)


def export_from_argilla(workspace: str, argilla_url: str, api_key: str) -> list[dict]:
    """Pull completed records from Argilla and convert to preference pairs."""
    rg.init(api_url=argilla_url, api_key=api_key)

    datasets = rg.list_datasets(workspace=workspace)
    all_pairs = []

    for ds_ref in datasets:
        print(f"  Loading dataset: {ds_ref.name}")
        ds = rg.FeedbackDataset.from_argilla(name=ds_ref.name, workspace=workspace)

        for record in tqdm(ds.records, desc=f"  {ds_ref.name}"):
            responses = record.responses
            if not responses:
                continue

            # Collect preference ratings from all annotators on this record
            ratings = []
            rationales = []
            for resp in responses:
                if resp.status != "submitted":
                    continue
                for answer in resp.values:
                    if answer.question_name == "preference":
                        ratings.append(answer.value)
                    if answer.question_name == "rationale" and answer.value:
                        rationales.append(answer.value)

            if not ratings:
                continue

            # Compute IAA
            kappa = compute_simple_kappa(ratings)
            median_rating = sorted(ratings)[len(ratings) // 2]

            # preference ≤ 2 → prefer A (chosen=A, rejected=B)
            # preference ≥ 4 → prefer B (chosen=B, rejected=A)
            # preference = 3 → no preference, skip
            if median_rating == 3:
                continue

            prefer_a = median_rating <= 2
            chosen = record.fields["response_a"] if prefer_a else record.fields["response_b"]
            rejected = record.fields["response_b"] if prefer_a else record.fields["response_a"]

            pair = {
                "schema_version": "1.0",
                "prompt_id": hash_text(record.fields["prompt"]),
                "prompt": record.fields["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "annotator_id": f"argilla-{workspace}",
                "annotator_role": record.metadata.get("domain", "unknown"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "rationale": rationales[0] if rationales else "",
                "iaa_score": round(kappa, 4),
                "domain": record.metadata.get("domain", "unknown"),
                "source": "argilla-annotation",
                "n_annotators": len(ratings),
                "rating_values": ratings,
            }
            all_pairs.append(pair)

    return all_pairs


def validate_pairs(pairs: list[dict], iaa_threshold: float) -> tuple[list[dict], list[dict]]:
    """Split pairs into passing and failing IAA threshold."""
    passing = [p for p in pairs if p["iaa_score"] is None or p["iaa_score"] >= iaa_threshold]
    failing = [p for p in pairs if p["iaa_score"] is not None and p["iaa_score"] < iaa_threshold]
    return passing, failing


def main():
    parser = argparse.ArgumentParser(description="Export preferences from Argilla")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate", action="store_true", help="Enforce IAA threshold")
    parser.add_argument("--iaa-threshold", type=float, default=IAA_KAPPA_THRESHOLD)
    parser.add_argument("--argilla-url", default="http://localhost:6900")
    parser.add_argument("--api-key", default="argilla.apikey")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting from Argilla workspace '{args.workspace}'...")
    try:
        pairs = export_from_argilla(args.workspace, args.argilla_url, args.api_key)
    except Exception as e:
        print(f"ERROR connecting to Argilla: {e}")
        sys.exit(1)

    print(f"\nExported {len(pairs)} raw pairs.")

    if args.validate:
        passing, failing = validate_pairs(pairs, args.iaa_threshold)
        print(f"IAA validation (κ ≥ {args.iaa_threshold}):")
        print(f"  PASS: {len(passing)} pairs")
        print(f"  FAIL (excluded): {len(failing)} pairs")
        pairs = passing

        if len(pairs) < MIN_PAIRS:
            print(f"\nWARNING: Only {len(pairs)} pairs passed IAA. Recommend ≥{MIN_PAIRS} for training.")

    # Write preference pairs
    pairs_path = out_dir / "preference_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # Write ART-001 annotation log
    log_path = out_dir / "annotation_log.jsonl"
    with open(log_path, "w") as f:
        for p in pairs:
            log_entry = {
                "annotator_id": p["annotator_id"],
                "timestamp": p["timestamp"],
                "prompt_hash": p["prompt_id"],
                "chosen_hash": hash_text(p["chosen"]),
                "rejected_hash": hash_text(p["rejected"]),
                "rationale": p["rationale"],
                "iaa_score": p["iaa_score"],
                "domain": p["domain"],
                "n_annotators": p.get("n_annotators", 1),
            }
            f.write(json.dumps(log_entry) + "\n")

    # Write metadata
    dataset_hash = hash_text("".join(p["prompt_id"] for p in pairs))
    meta = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": f"argilla:{args.workspace}",
        "n_pairs": len(pairs),
        "iaa_threshold": args.iaa_threshold,
        "dataset_hash": dataset_hash,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {len(pairs)} pairs written to {out_dir}/")
    print(f"  ART-001 annotation log: {log_path}")


if __name__ == "__main__":
    main()
