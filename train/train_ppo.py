"""
train/train_ppo.py

PPO-based RLHF fine-tuning — advanced configuration.
Use when DPO is insufficient and iterative reward shaping is required.

Requires a trained reward model (from train_reward_model.py).

Usage:
    python train/train_ppo.py \
        --config configs/ppo_advanced.yaml \
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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(dataset_dir: str) -> list[str]:
    prompts = []
    with open(Path(dataset_dir) / "preference_pairs.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return list(set(prompts))  # deduplicate


def main():
    parser = argparse.ArgumentParser(description="PPO RLHF fine-tuning for AlignR")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--mlflow-uri",   default="http://localhost:5000")
    args = parser.parse_args()

    config   = load_config(args.config)
    ppo_cfg  = config["ppo"]
    lora_cfg = config["lora"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if ppo_cfg.get("dtype") == "bfloat16" else torch.float32

    print("Loading prompts...")
    prompts = load_prompts(args.dataset)
    print(f"  {len(prompts)} unique prompts.")

    meta_path = Path(args.dataset) / "metadata.json"
    dataset_hash = "unknown"
    if meta_path.exists():
        with open(meta_path) as f:
            dataset_hash = json.load(f).get("dataset_hash", "unknown")

    print(f"Loading base model: {config['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Policy model with value head for PPO
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config["base_model"],
        torch_dtype=dtype,
        device_map="auto",
        peft_config=lora_config,
    )

    # Reference model (frozen)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config["base_model"],
        torch_dtype=dtype,
        device_map="auto",
    )
    for param in ref_model.parameters():
        param.requires_grad = False

    # Reward model
    print(f"Loading reward model from {args.reward_model}...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        num_labels=1,
        torch_dtype=dtype,
        device_map="auto",
    )
    reward_model.eval()

    # Reward pipeline
    reward_pipe = pipeline(
        "text-classification",
        model=reward_model,
        tokenizer=tokenizer,
        device_map="auto",
        truncation=True,
        max_length=ppo_cfg.get("max_length", 1024),
    )

    # PPO config
    ppo_config = PPOConfig(
        model_name=config["base_model"],
        learning_rate=ppo_cfg["learning_rate"],
        batch_size=ppo_cfg["batch_size"],
        mini_batch_size=ppo_cfg["batch_size"],
        gradient_accumulation_steps=ppo_cfg["gradient_accumulation"],
        ppo_epochs=ppo_cfg["ppo_epochs"],
        kl_penalty=ppo_cfg.get("kl_penalty", "kl"),
        init_kl_coef=ppo_cfg.get("kl_penalty_beta", 0.05),
        target_kl=ppo_cfg.get("target_kl", 0.1),
        cliprange=ppo_cfg.get("ppo_clip_ratio", 0.2),
        vf_coef=ppo_cfg.get("vf_coef", 0.1),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        seed=ppo_cfg["seed"],
        log_with=None,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("alignr-ppo-finetune")
    with mlflow.start_run(run_name=f"ppo-{int(time.time())}") as run:
        mlflow.log_params({
            "base_model":           config["base_model"],
            "strategy":             "ppo",
            "kl_penalty_beta":      ppo_cfg.get("kl_penalty_beta", 0.05),
            "ppo_clip_ratio":       ppo_cfg.get("ppo_clip_ratio", 0.2),
            "learning_rate":        ppo_cfg["learning_rate"],
            "lora_rank":            lora_cfg["r"],
            "ppo_epochs":           ppo_cfg["ppo_epochs"],
            "seed":                 ppo_cfg["seed"],
            "n_prompts":            len(prompts),
            "dataset_hash":         dataset_hash,
        })

        ds = Dataset.from_dict({"prompt": prompts})
        prompt_tensors = [
            tokenizer.encode(p, return_tensors="pt").squeeze()
            for p in prompts[:ppo_cfg["batch_size"] * 50]  # cap for demo
        ]

        print(f"\nPPO training — {len(prompt_tensors)} prompts, "
              f"{ppo_cfg['ppo_epochs']} PPO epochs per batch...")

        step = 0
        for i in range(0, len(prompt_tensors), ppo_cfg["batch_size"]):
            batch_prompts = prompt_tensors[i:i + ppo_cfg["batch_size"]]
            if not batch_prompts:
                break

            # Generate responses from policy
            response_tensors = trainer.generate(
                batch_prompts,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode for reward scoring
            texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            rewards_raw = reward_pipe(texts, truncation=True)
            rewards = [torch.tensor(r["score"]) for r in rewards_raw]

            # PPO step
            stats = trainer.step(batch_prompts, response_tensors, rewards)

            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            mlflow.log_metrics({
                "mean_reward": mean_reward,
                "ppo_step": step,
            }, step=step)

            if step % 10 == 0:
                print(f"  Step {step}: mean reward = {mean_reward:.4f}")
            step += 1

        # Save
        trainer.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mlflow.log_artifacts(str(output_dir), artifact_path="aligned_model_lora_ppo")
        mlflow.log_param("mlflow_run_id", run.info.run_id)

        print(f"\nPPO adapter saved to {output_dir}/")
        print(f"MLflow run ID: {run.info.run_id}  ← save this for artefact bundle")
        print("Next: python train/merge_lora.py")


if __name__ == "__main__":
    main()
