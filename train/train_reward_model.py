
import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs(dataset_dir):
    pairs = []
    with open(Path(dataset_dir) / "preference_pairs.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--mlflow-uri", default="./mlflow_runs")
    args = parser.parse_args()

    config     = load_config(args.config)
    rm_cfg     = config["reward_model"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading preference pairs...")
    pairs = load_pairs(args.dataset)
    print(f"  {len(pairs)} pairs loaded.")

    meta_path    = Path(args.dataset) / "metadata.json"
    dataset_hash = "unknown"
    if meta_path.exists():
        with open(meta_path) as f:
            dataset_hash = json.load(f).get("dataset_hash", "unknown")

    MODEL_NAME = "distilbert-base-uncased"
    print(f"Loading base model: {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # distilbert has no EOS token — set it explicitly
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "[SEP]"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1
    )
    model.resize_token_embeddings(len(tokenizer))

    ds = Dataset.from_list([
        {
            "chosen":   p["prompt"] + " " + p["chosen"],
            "rejected": p["prompt"] + " " + p["rejected"],
        }
        for p in pairs
    ])
    val_size = max(1, int(len(ds) * 0.1))
    ds_split = ds.train_test_split(test_size=val_size, seed=42)
    print(f"  Train: {len(ds_split['train'])} | Val: {len(ds_split['test'])}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("alignr-reward-model")

    with mlflow.start_run(run_name=f"reward-model-{int(time.time())}") as run:
        mlflow.log_params({
            "base_model":           MODEL_NAME,
            "rm_epochs":            rm_cfg["epochs"],
            "learning_rate":        rm_cfg["learning_rate"],
            "n_pairs":              len(pairs),
            "dataset_hash":         dataset_hash,
            "seed":                 42,
            "validation_threshold": rm_cfg["validation_threshold"],
        })

        training_args = RewardConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=rm_cfg["epochs"],
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=rm_cfg["learning_rate"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            seed=42,
            report_to="none",
            logging_steps=20,
            max_length=256,
        )

        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_split["train"],
            eval_dataset=ds_split["test"],
            processing_class=tokenizer,
        )

        print("\nTraining reward model...")
        trainer.train()

        correct = 0
        total   = min(200, len(ds_split["test"]))
        model.eval()
        device = next(model.parameters()).device
        for i in range(total):
            item  = ds_split["test"][i]
            enc_c = tokenizer(item["chosen"],   return_tensors="pt",
                              truncation=True, max_length=256, padding=True).to(device)
            enc_r = tokenizer(item["rejected"], return_tensors="pt",
                              truncation=True, max_length=256, padding=True).to(device)
            with torch.no_grad():
                sc = model(**enc_c).logits.squeeze().item()
                sr = model(**enc_r).logits.squeeze().item()
            if sc > sr:
                correct += 1

        val_accuracy = correct / total
        mlflow.log_metrics({"val_accuracy": val_accuracy})

        threshold = rm_cfg["validation_threshold"]
        print(f"\nValidation accuracy: {val_accuracy:.1%}")
        print(f"Threshold:           {threshold:.0%}")

        if val_accuracy < threshold:
            print("FAIL: Below threshold. Pipeline aborted.")
            mlflow.log_param("gate_status", "FAIL")
            sys.exit(1)

        print("PASS: Saving reward model...")
        mlflow.log_param("gate_status", "PASS")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"\nReward model saved to {output_dir}/")
        print(f"MLflow run ID: {run.info.run_id}")
        print("\nART-002 complete.")


if __name__ == "__main__":
    main()
