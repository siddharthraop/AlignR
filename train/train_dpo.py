
import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


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
    parser.add_argument("--config",      required=True)
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--mlflow-uri",  default="./mlflow_runs")
    args = parser.parse_args()

    config     = load_config(args.config)
    dpo_cfg    = config["dpo"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading preference pairs...")
    pairs = load_pairs(args.dataset)
    print(f"  {len(pairs)} pairs loaded.")

    MODEL_NAME = "distilgpt2"
    print(f"Loading base model: {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    ds = Dataset.from_list([
        {
            "prompt":   p["prompt"],
            "chosen":   p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])
    val_size = max(1, int(len(ds) * 0.1))
    ds_split = ds.train_test_split(test_size=val_size, seed=42)
    print(f"  Train: {len(ds_split['train'])} | Val: {len(ds_split['test'])}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("alignr-dpo")

    with mlflow.start_run(run_name=f"dpo-{int(time.time())}") as run:
        mlflow.log_params({
            "base_model":    MODEL_NAME,
            "dpo_epochs":    dpo_cfg["epochs"],
            "learning_rate": dpo_cfg["learning_rate"],
            "beta":          dpo_cfg["beta"],
            "n_pairs":       len(pairs),
            "seed":          dpo_cfg["seed"],
        })

        training_args = DPOConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=dpo_cfg["epochs"],
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=dpo_cfg["learning_rate"],
            beta=dpo_cfg["beta"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            seed=dpo_cfg["seed"],
            report_to="none",
            logging_steps=20,
            max_length=256,
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_split["train"],
            eval_dataset=ds_split["test"],
            processing_class=tokenizer,
        )

        print("\nRunning DPO fine-tuning...")
        trainer.train()

        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss":           eval_results.get("eval_loss", 0),
            "eval_rewards/chosen": eval_results.get("eval_rewards/chosen", 0),
        })

        print("\nSaving aligned model...")
        mlflow.log_param("status", "PASS")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"\nAligned model saved to {output_dir}/")
        print(f"MLflow run ID: {run.info.run_id}")
        print("\nART-003 and ART-004 complete.")


if __name__ == "__main__":
    main()
