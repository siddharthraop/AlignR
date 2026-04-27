"""
artefacts/generate_manifest.py

Generates the AlignR compliance artefact bundle manifest.
Collects ART-001 through ART-006, computes SHA-256 hashes,
and writes the manifest JSON.

Usage:
    python artefacts/generate_manifest.py \
        --model-version alignr-v1.0.0 \
        --mlflow-run-id <run_id> \
        --output artefacts/bundle/
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file or directory (directory → hash of file listing)."""
    h = hashlib.sha256()
    if path.is_file():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    elif path.is_dir():
        # Hash the sorted list of files + their sizes as a proxy
        file_list = sorted(str(p) + str(p.stat().st_size)
                           for p in path.rglob("*") if p.is_file())
        h.update("\n".join(file_list).encode())
    else:
        return "missing"
    return h.hexdigest()[:16]


def find_artefact(paths: list[str]) -> tuple[Path | None, str]:
    """Return the first existing path from a list of candidates."""
    for p in paths:
        path = Path(p)
        if path.exists():
            return path, sha256_file(path)
    return None, "missing"


def artefact_entry(art_type: str, candidates: list[str]) -> dict:
    path, sha = find_artefact(candidates)
    status = "complete" if path else "missing"
    return {
        "type":   art_type,
        "path":   str(path) if path else "NOT FOUND",
        "sha256": sha,
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate AlignR artefact bundle manifest")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--dataset-dir",   default="data/pref_dataset/")
    parser.add_argument("--reward-model",  default="./reward_model/")
    parser.add_argument("--lora-dir",      default="./aligned_model_lora/")
    parser.add_argument("--aligned-model", default="./aligned_model/")
    parser.add_argument("--redteam-report",default="eval/redteam_report.pdf")
    parser.add_argument("--cco-signoff",   default=None,
                        help="Path to signed CCO PDF (ART-006). Leave blank if pending.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating artefact bundle manifest for {args.model_version}...")

    artefacts = {
        "ART-001": artefact_entry(
            "annotation_log",
            [f"{args.dataset_dir}/annotation_log.jsonl"]
        ),
        "ART-002": artefact_entry(
            "reward_model_weights",
            [args.reward_model]
        ),
        "ART-003": artefact_entry(
            "finetuning_run_record",
            [args.lora_dir, f"{args.lora_dir}/adapter_config.json"]
        ),
        "ART-004": artefact_entry(
            "aligned_model_weights",
            [args.aligned_model]
        ),
        "ART-005": artefact_entry(
            "redteam_report",
            [args.redteam_report]
        ),
        "ART-006": artefact_entry(
            "cco_signoff",
            [args.cco_signoff] if args.cco_signoff else []
        ),
    }

    # Determine overall gate status
    required_complete = ["ART-001", "ART-002", "ART-003", "ART-004", "ART-005"]
    all_required_complete = all(
        artefacts[a]["status"] == "complete" for a in required_complete
    )
    cco_complete = artefacts["ART-006"]["status"] == "complete"
    release_gate_passed = all_required_complete and cco_complete

    manifest = {
        "schema_version":      "1.0",
        "model_version":       args.model_version,
        "created_at":          datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id":       args.mlflow_run_id,
        "deployment_scope":    "pending CCO sign-off",
        "release_gate_passed": release_gate_passed,
        "next_review_date":    None,
        "artefacts":           artefacts,
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print status table
    print(f"\nArtefact bundle status — {args.model_version}")
    print(f"{'ID':<10} {'Type':<30} {'Status':<12} {'SHA256'}")
    print("-" * 70)
    for art_id, entry in artefacts.items():
        status_label = "✓ complete" if entry["status"] == "complete" else "✗ MISSING"
        print(f"{art_id:<10} {entry['type']:<30} {status_label:<12} {entry['sha256']}")

    print(f"\nRelease gate: {'PASSED ✓' if release_gate_passed else 'NOT PASSED'}")
    if not cco_complete:
        print("  ART-006 pending — obtain CCO/DPO sign-off and re-run with --cco-signoff <path>")

    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
