"""
eval/score_reward_model.py

Standalone validation of the reward model on a held-out preference set.
Used to spot-check ART-002 after training.

Usage:
    python eval/score_reward_model.py \
        --reward-model ./reward_model/ \
        --dataset data/pref_dataset/
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Validate reward model on held-out pairs")
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--n",            type=int, default=500,
                        help="Number of held-out pairs to evaluate")
    parser.add_argument("--max-length",   type=int, default=512)
    args = parser.parse_args()

    # Load pairs — use last n as held-out
    pairs = []
    with open(Path(args.dataset) / "preference_pairs.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    pairs = pairs[-args.n:]
    print(f"Evaluating on {len(pairs)} held-out pairs...")

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, num_labels=1, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    correct = 0
    for pair in tqdm(pairs):
        enc_chosen = tokenizer(
            pair["prompt"], pair["chosen"],
            return_tensors="pt", truncation=True,
            max_length=args.max_length, padding="max_length"
        ).to(model.device)
        enc_rejected = tokenizer(
            pair["prompt"], pair["rejected"],
            return_tensors="pt", truncation=True,
            max_length=args.max_length, padding="max_length"
        ).to(model.device)

        with torch.no_grad():
            score_chosen   = model(**enc_chosen).logits.squeeze().item()
            score_rejected = model(**enc_rejected).logits.squeeze().item()

        if score_chosen > score_rejected:
            correct += 1

    accuracy = correct / len(pairs)
    print(f"\nReward model accuracy: {accuracy:.1%} ({correct}/{len(pairs)})")
    print(f"Hard gate threshold:   80.0%")
    print(f"Status: {'PASS ✓' if accuracy >= 0.80 else 'FAIL ✗'}")


if __name__ == "__main__":
    main()
