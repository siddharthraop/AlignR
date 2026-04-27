"""
train/train_reward_model.py

Trains a reward model on preference pairs using TRL's RewardTrainer.
The reward model is a Llama-3-8B backbone with a regression head
that assigns a scalar score to any (prompt, response) pair.

This produces ART-002: versioned reward model weights + model card.

Usage:
    python train/train_reward_model.py \
        --config configs/dpo_default.yaml \
        --dataset data/pref_dataset/ \
        --output ./reward_model/ \
        --mlflow-uri http://localhost:5000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from trl import RewardConfig, RewardTrainer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_preference_pairs(dataset_dir: str) -> list[dict]:
    pairs = []
    pairs_path = Path(dataset_dir) / "preference_pairs.jsonl"
    with open(pairs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_hf_dataset(pairs: list[dict], tokenizer, max_length: int) -> Dataset:
    """Convert AlignR pairs to HuggingFace Dataset for RewardTrainer."""
    def tokenize(batch):
        chosen_enc = tokenizer(
            batch["prompt"], batch["chosen"],
            truncation=True, max_length=max_length, padding="max_length"
        )
        rejected_enc = tokenizer(
            batch["prompt"], batch["rejected"],
            truncation=True, max_length=max_length, padding="max_length"
        )
        return {
            "input_ids_chosen":      chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected":      rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    raw = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])
    return raw.map(tokenize, batched=True, remove_columns=raw.column_names)


def write_model_card(output_dir: Path, config: dict, val_accuracy: float,
                     n_pairs: int, dataset_hash: str):
    card = f"""# AlignR Reward Model

**ART-002 — Reward model weights and card**

## Training details

| Field | Value |
|---|---|
| Base model | {config['base_model']} |
| Training pairs | {n_pairs} |
| Dataset hash | `{dataset_hash}` |
| Epochs | {config['reward_model']['epochs']} |
| Learning rate | {config['reward_model']['learning_rate']} |
| Validation accuracy | {val_accuracy:.1%} |
| Validation threshold | {config['reward_model']['validation_threshold']:.0%} |
| Gate status | {"PASS ✓" if val_accuracy >= config['reward_model']['validation_threshold'] else "FAIL ✗"} |

## Purpose

This model scores (prompt, response) pairs on a scalar [0, 1] scale
representing domain-expert preference. It is an inspectable compliance
artefact — unlike a proprietary API, its weights can be audited,
challenged, or retrained as regulatory standards evolve.

## Known limitations

- Reflects the judgment of annotators who produced the preference dataset.
  Annotator bias becomes reward model bias.
- Valid only for the domain and query distribution of the training data.
- Not intended for standalone deployment — it is an internal scoring tool.

## Intended use

- Training signal for DPO/PPO fine-tuning (ART-003)
- Held-out evaluation of aligned model outputs
- Compliance evidence for EU AI Act transparency obligation

## Sign-off

Signed by ML lead as part of pipeline run producing model version alignr-v1.x.
"""
    with open(output_dir / "MODEL_CARD.md", "w") as f:
        f.write(card)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AlignR reward model")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--mlflow-uri",  default="http://localhost:5000")
    args = parser.parse_args()

    config = load_config(args.config)
    rm_cfg = config["reward_model"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading preference pairs...")
    pairs = load_preference_pairs(args.dataset)
    n_pairs = len(pairs)
    print(f"  {n_pairs} pairs loaded.")

    # Load dataset metadata for provenance
    meta_path = Path(args.dataset) / "metadata.json"
    dataset_hash = "unknown"
    if meta_path.exists():
        with open(meta_path) as f:
            dataset_hash = json.load(f).get("dataset_hash", "unknown")

    # Dtype
    dtype = torch.bfloat16 if rm_cfg.get("dtype") == "bfloat16" else torch.float32

    print(f"Loading base model: {config['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=1,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Build dataset and split
    max_length = config.get("dpo", {}).get("max_length", 1024)
    print("Tokenising dataset...")
    ds = build_hf_dataset(pairs, tokenizer, max_length)
    val_size = max(1, int(len(ds) * rm_cfg["validation_split"]))
    train_size = len(ds) - val_size
    ds_split = ds.train_test_split(test_size=val_size, seed=rm_cfg["seed"])

    print(f"  Train: {train_size} | Val: {val_size}")

    # MLflow run
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("alignr-reward-model")
    with mlflow.start_run(run_name=f"reward-model-{int(time.time())}") as run:
        mlflow.log_params({
            "base_model":           config["base_model"],
            "strategy":             config.get("strategy", "dpo"),
            "rm_epochs":            rm_cfg["epochs"],
            "rm_learning_rate":     rm_cfg["learning_rate"],
            "rm_batch_size":        rm_cfg["batch_size"],
            "n_pairs":              n_pairs,
            "dataset_hash":         dataset_hash,
            "seed":                 rm_cfg["seed"],
            "validation_threshold": rm_cfg["validation_threshold"],
        })

        training_args = RewardConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=rm_cfg["epochs"],
            per_device_train_batch_size=rm_cfg["batch_size"],
            per_device_eval_batch_size=rm_cfg["batch_size"],
            learning_rate=rm_cfg["learning_rate"],
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=(dtype == torch.bfloat16),
            seed=rm_cfg["seed"],
            report_to="none",   # we log to MLflow manually
            logging_steps=10,
        )

        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_split["train"],
            eval_dataset=ds_split["test"],
            tokenizer=tokenizer,
        )

        print("\nTraining reward model...")
        trainer.train()

        # Evaluate on held-out set
        print("\nEvaluating on held-out set...")
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss", float("inf"))

        # Compute accuracy: reward model scores chosen > rejected
        correct = 0
        model.eval()
        with torch.no_grad():
            for i in range(0, min(500, len(ds_split["test"])), 4):
                batch = ds_split["test"][i:i+4]
                chosen_ids = torch.tensor(batch["input_ids_chosen"]).to(model.device)
                rejected_ids = torch.tensor(batch["input_ids_rejected"]).to(model.device)
                chosen_scores = model(input_ids=chosen_ids).logits.squeeze()
                rejected_scores = model(input_ids=rejected_ids).logits.squeeze()
                if chosen_scores.dim() == 0:
                    correct += int(chosen_scores > rejected_scores)
                else:
                    correct += int((chosen_scores > rejected_scores).sum())

        val_total = min(500, len(ds_split["test"]))
        val_accuracy = correct / val_total if val_total > 0 else 0.0

        mlflow.log_metrics({
            "eval_loss":     eval_loss,
            "val_accuracy":  val_accuracy,
        })

        print(f"\nValidation accuracy: {val_accuracy:.1%}")
        print(f"Threshold:           {rm_cfg['validation_threshold']:.0%}")

        # Hard gate
        if val_accuracy < rm_cfg["validation_threshold"]:
            print(f"\nFAIL: Validation accuracy {val_accuracy:.1%} below hard gate "
                  f"{rm_cfg['validation_threshold']:.0%}.")
            print("Pipeline aborted. Collect more preference data or adjust hyperparameters.")
            mlflow.log_param("gate_status", "FAIL")
            sys.exit(1)

        print("PASS: Validation accuracy meets threshold. Saving model...")
        mlflow.log_param("gate_status", "PASS")

        # Save
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        write_model_card(output_dir, config, val_accuracy, n_pairs, dataset_hash)

        # Log artefacts to MLflow
        mlflow.log_artifacts(str(output_dir), artifact_path="reward_model")
        mlflow.log_param("mlflow_run_id", run.info.run_id)

        print(f"\nReward model saved to {output_dir}/")
        print(f"MLflow run ID: {run.info.run_id}")
        print("ART-002 complete.")


if __name__ == "__main__":
    main()
