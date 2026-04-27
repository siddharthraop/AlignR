"""
train/train_dpo.py

Fine-tunes Llama-3-8B-Instruct using Direct Preference Optimisation (DPO)
with LoRA adapters via PEFT. Checkpointing is enabled by default.

This produces ART-003: MLflow run record (config, loss curve, seed, dataset hash).
The aligned LoRA adapter is the input to merge_lora.py (ART-004).

Usage:
    python train/train_dpo.py \
        --config configs/dpo_default.yaml \
        --dataset data/pref_dataset/ \
        --reward-model ./reward_model/ \
        --output ./aligned_model_lora/ \
        --mlflow-uri http://localhost:5000
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs(dataset_dir: str) -> list[dict]:
    pairs = []
    with open(Path(dataset_dir) / "preference_pairs.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_dpo_dataset(pairs: list[dict]) -> Dataset:
    """Build a HuggingFace Dataset in the format DPOTrainer expects."""
    return Dataset.from_list([
        {
            "prompt":   p["prompt"],
            "chosen":   p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])


def build_lora_config(lora_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for AlignR")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--mlflow-uri",   default="http://localhost:5000")
    args = parser.parse_args()

    config = load_config(args.config)
    dpo_cfg  = config["dpo"]
    lora_cfg = config["lora"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading preference pairs...")
    pairs = load_pairs(args.dataset)
    print(f"  {len(pairs)} pairs loaded.")

    meta_path = Path(args.dataset) / "metadata.json"
    dataset_hash = "unknown"
    if meta_path.exists():
        with open(meta_path) as f:
            dataset_hash = json.load(f).get("dataset_hash", "unknown")

    ds = build_dpo_dataset(pairs)
    val_size  = max(1, int(len(ds) * 0.05))
    ds_split  = ds.train_test_split(test_size=val_size, seed=dpo_cfg["seed"])

    # Dtype / quantisation
    use_qlora = config.get("quantisation", {}).get("load_in_4bit", False)
    dtype = torch.bfloat16 if dpo_cfg.get("dtype") == "bfloat16" else torch.float32

    bnb_config = None
    if use_qlora:
        print("QLoRA mode: loading 4-bit quantised base model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=config["quantisation"].get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=config["quantisation"].get("bnb_4bit_quant_type", "nf4"),
        )

    print(f"Loading base model: {config['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # Reference model (frozen copy of base — DPO KL anchor)
    model_ref = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=bnb_config,
    )
    for param in model_ref.parameters():
        param.requires_grad = False

    # Apply LoRA
    lora_config = build_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("alignr-dpo-finetune")
    with mlflow.start_run(run_name=f"dpo-{int(time.time())}") as run:
        mlflow.log_params({
            "base_model":           config["base_model"],
            "strategy":             "dpo",
            "dpo_beta":             dpo_cfg["beta"],
            "learning_rate":        dpo_cfg["learning_rate"],
            "lora_rank":            lora_cfg["r"],
            "lora_alpha":           lora_cfg["alpha"],
            "epochs":               dpo_cfg["epochs"],
            "batch_size":           dpo_cfg["batch_size"],
            "gradient_accumulation":dpo_cfg["gradient_accumulation"],
            "seed":                 dpo_cfg["seed"],
            "n_pairs":              len(pairs),
            "dataset_hash":         dataset_hash,
            "dtype":                dpo_cfg.get("dtype", "bfloat16"),
            "use_qlora":            use_qlora,
        })

        training_args = DPOConfig(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=dpo_cfg["epochs"],
            per_device_train_batch_size=dpo_cfg["batch_size"],
            per_device_eval_batch_size=dpo_cfg["batch_size"],
            gradient_accumulation_steps=dpo_cfg["gradient_accumulation"],
            learning_rate=dpo_cfg["learning_rate"],
            warmup_ratio=dpo_cfg.get("warmup_ratio", 0.1),
            beta=dpo_cfg["beta"],
            max_length=dpo_cfg.get("max_length", 1024),
            max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=False,  # LoRA checkpointing
            bf16=(dtype == torch.bfloat16),
            seed=dpo_cfg["seed"],
            report_to="none",
            logging_steps=10,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=model_ref,
            args=training_args,
            train_dataset=ds_split["train"],
            eval_dataset=ds_split["test"],
            tokenizer=tokenizer,
        )

        print(f"\nDPO fine-tuning — {len(pairs)} pairs, {dpo_cfg['epochs']} epochs...")
        print(f"Checkpoints saved every 100 steps. Safe to resume if Colab times out.")
        trainer.train()

        # Save LoRA adapter
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Log artefacts
        mlflow.log_artifacts(str(output_dir), artifact_path="aligned_model_lora")
        mlflow.log_param("mlflow_run_id", run.info.run_id)

        print(f"\nLoRA adapter saved to {output_dir}/")
        print(f"MLflow run ID: {run.info.run_id}  ← save this for artefact bundle")
        print("ART-003 complete. Next: python train/merge_lora.py")


if __name__ == "__main__":
    main()
