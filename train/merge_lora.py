"""
train/merge_lora.py

Merges a LoRA adapter into the base model weights and exports
the result as safetensors. Runs on CPU — no GPU required (~5 min).

This produces ART-004: aligned model weights + model card.

Usage:
    python train/merge_lora.py \
        --base meta-llama/Llama-3-8B-Instruct \
        --lora ./aligned_model_lora/ \
        --output ./aligned_model/
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_model_card(output_dir: Path, base_model: str, lora_dir: str):
    card = f"""# AlignR Aligned Model

**ART-004 — Aligned model weights and card**

## Model details

| Field | Value |
|---|---|
| Base model | {base_model} |
| LoRA adapter | {lora_dir} |
| Merge date | {datetime.now(timezone.utc).isoformat()} |
| Format | safetensors |
| Merge method | `merge_and_unload()` — full weight merge, no adapter dependency |

## Alignment objectives

This model has been fine-tuned using Direct Preference Optimisation (DPO)
on domain-expert preference data. The training signal encodes institutional
judgment about appropriate response behaviour in regulated enterprise contexts.

## Intended deployment scope

Internal-facing only. See CCO sign-off document (ART-006) for permitted
deployment scope and restrictions.

## Known limitations

- Effective only within the domain and query distribution of the training data.
- Reward hacking possible if reward model has blind spots — see ART-005 red-team report.
- LoRA r=16 may be insufficient for highly domain-specific objectives.
  Upgrade path: increase LoRA rank or move to full fine-tuning.

## Compliance

This model card is part of the AlignR compliance artefact bundle (ART-004).
Deployment requires CCO/DPO sign-off (ART-006). See ART-005 for red-team results.

## Deployment restriction

**This model may not be deployed to production without a signed CCO/DPO sign-off
document (ART-006) referencing this exact model version.**
"""
    with open(output_dir / "MODEL_CARD.md", "w") as f:
        f.write(card)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base",   required=True, help="Base model name or path")
    parser.add_argument("--lora",   required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base}")
    print("(Running on CPU — this takes ~5 minutes and requires ~16GB RAM)")

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    # Load in float16 for merge — bfloat16 can cause issues on CPU merge
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter from {args.lora}...")
    model = PeftModel.from_pretrained(base_model, args.lora, device_map="cpu")

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}/...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    write_model_card(output_dir, args.base, args.lora)

    print(f"\nDone. Merged model saved to {output_dir}/")
    print("ART-004 complete.")
    print("\nNext:")
    print("  python eval/run_eval.py --model ./aligned_model/ ...")
    print("  python eval/run_redteam.py --model ./aligned_model/ ...")


if __name__ == "__main__":
    main()
