"""
data/bootstrap/hh_rlhf_prep.py

Downloads Anthropic HH-RLHF from HuggingFace Hub and converts it
to the AlignR preference pair format for MVP bootstrapping.

Usage:
    python data/bootstrap/hh_rlhf_prep.py \
        --output data/pref_dataset/ \
        --n_pairs 5000
"""

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


ALIGNR_SCHEMA_VERSION = "1.0"


def parse_hh_conversation(text: str) -> str:
    """Extract the final human turn from an HH-RLHF conversation string."""
    lines = text.strip().split("\n\n")
    human_turns = [l.replace("Human: ", "").strip() for l in lines if l.startswith("Human:")]
    return human_turns[-1] if human_turns else text[:512]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def convert_pair(row: dict) -> dict:
    """Convert one HH-RLHF row to AlignR preference pair schema."""
    prompt = parse_hh_conversation(row["chosen"])
    chosen_response = row["chosen"].split("Assistant:")[-1].strip()
    rejected_response = row["rejected"].split("Assistant:")[-1].strip()

    return {
        "schema_version": ALIGNR_SCHEMA_VERSION,
        "prompt_id": hash_text(prompt),
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "annotator_id": "bootstrap-hh-rlhf",
        "annotator_role": "bootstrap",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rationale": "Bootstrapped from Anthropic HH-RLHF. Replace with domain-expert annotation for production.",
        "iaa_score": None,          # not applicable for bootstrap
        "domain": "general",
        "source": "anthropic/hh-rlhf",
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare HH-RLHF bootstrap dataset")
    parser.add_argument("--output", required=True, help="Output directory for preference dataset")
    parser.add_argument("--n_pairs", type=int, default=5000, help="Number of pairs to extract")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Anthropic HH-RLHF from HuggingFace Hub...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    # Sample n_pairs rows
    indices = random.sample(range(len(ds)), min(args.n_pairs, len(ds)))
    sampled = ds.select(indices)

    pairs = []
    annotation_log = []

    print(f"Converting {len(sampled)} pairs to AlignR format...")
    for row in tqdm(sampled):
        pair = convert_pair(row)
        pairs.append(pair)

        # Produce ART-001 annotation log entry
        annotation_log.append({
            "annotator_id": pair["annotator_id"],
            "timestamp": pair["timestamp"],
            "prompt_hash": pair["prompt_id"],
            "chosen_hash": hash_text(pair["chosen"]),
            "rejected_hash": hash_text(pair["rejected"]),
            "rationale": pair["rationale"],
            "iaa_score": pair["iaa_score"],
            "domain": pair["domain"],
            "source": pair["source"],
        })

    # Write preference pairs (TRL-compatible JSONL)
    pairs_path = out_dir / "preference_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # Write annotation log (ART-001)
    log_path = out_dir / "annotation_log.jsonl"
    with open(log_path, "w") as f:
        for entry in annotation_log:
            f.write(json.dumps(entry) + "\n")

    # Write dataset metadata
    meta = {
        "schema_version": ALIGNR_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "anthropic/hh-rlhf",
        "n_pairs": len(pairs),
        "seed": args.seed,
        "note": "Bootstrap dataset for MVP. Replace with domain-expert annotation for production.",
        "dataset_hash": hash_text("".join(p["prompt_id"] for p in pairs)),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {len(pairs)} preference pairs written to {out_dir}/")
    print(f"  preference_pairs.jsonl  — TRL-compatible training data")
    print(f"  annotation_log.jsonl    — ART-001 compliance artefact")
    print(f"  metadata.json           — dataset provenance record")
    print(f"\nNOTE: This is bootstrap data only. For production, run a domain-expert")
    print(f"annotation campaign via Argilla and export with data/export_preferences.py")


if __name__ == "__main__":
    main()
