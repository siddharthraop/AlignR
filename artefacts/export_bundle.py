"""
artefacts/export_bundle.py

Exports the complete AlignR compliance artefact bundle as a ZIP file
for regulator or internal audit submission.

Usage:
    python artefacts/export_bundle.py \
        --manifest artefacts/bundle/manifest.json \
        --output artefacts/alignr-v1.0.0-bundle.zip
"""

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export AlignR artefact bundle as ZIP")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output",   required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_version = manifest.get("model_version", "unknown")
    print(f"Exporting artefact bundle for {model_version}...")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Always include the manifest itself
        zf.write(manifest_path, "manifest.json")

        for art_id, entry in manifest["artefacts"].items():
            if entry["status"] != "complete":
                print(f"  {art_id}: SKIPPED (status={entry['status']})")
                continue

            path = Path(entry["path"])
            if not path.exists():
                print(f"  {art_id}: SKIPPED (path not found: {path})")
                continue

            if path.is_file():
                arcname = f"{art_id}/{path.name}"
                zf.write(path, arcname)
                print(f"  {art_id}: {arcname}")
            elif path.is_dir():
                # For directories (model weights), include all files
                for file_path in sorted(path.rglob("*")):
                    if file_path.is_file():
                        arcname = f"{art_id}/{file_path.relative_to(path.parent)}"
                        zf.write(file_path, arcname)
                print(f"  {art_id}: {path}/ (directory, {sum(1 for _ in path.rglob('*') if _.is_file())} files)")

        # Include export metadata
        export_meta = {
            "exported_at":    datetime.now(timezone.utc).isoformat(),
            "model_version":  model_version,
            "mlflow_run_id":  manifest.get("mlflow_run_id"),
            "release_gate":   manifest.get("release_gate_passed"),
            "note": "AlignR compliance artefact bundle. All artefacts are immutable after model release."
        }
        zf.writestr("EXPORT_METADATA.json", json.dumps(export_meta, indent=2))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nBundle exported: {output_path} ({size_mb:.1f} MB)")

    if not manifest.get("release_gate_passed"):
        print("\nWARNING: release_gate_passed = false in manifest.")
        print("ART-006 (CCO sign-off) is pending. This bundle is incomplete for deployment.")


if __name__ == "__main__":
    main()
