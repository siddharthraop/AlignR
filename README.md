# AlignR

**RLHF fine-tuning pipeline for regulated enterprise AI.** Takes an open-weight foundation model and produces a behaviourally aligned, enterprise-deployable model using preference data collected from domain experts — compliance officers, underwriters, legal counsel, claims adjusters. Every stage of the pipeline produces structured compliance artefacts that map directly to EU AI Act conformity assessment requirements.

> **Design doc →** [raosiddharthp.github.io/AlignR](https://raosiddharthp.github.io/AlignR)

---

## Repo structure

```
alignr/
├── configs/
│   ├── dpo_default.yaml            # DPO · Llama-3-8B · LoRA r=16 · recommended
│   ├── dpo_conservative.yaml       # DPO · β=0.2 · healthcare / high-stakes
│   ├── ppo_advanced.yaml           # PPO full loop · iterative reward shaping
│   └── qlora_t4.yaml               # QLoRA fallback · Kaggle T4 16GB
├── data/
│   ├── argilla_setup.py            # Argilla workspace + dataset config
│   ├── export_preferences.py       # Argilla SDK → HuggingFace Dataset
│   ├── validate_dataset.py         # schema validation · IAA threshold check
│   └── bootstrap/
│       └── hh_rlhf_prep.py         # Anthropic HH-RLHF → AlignR format (MVP seed)
├── train/
│   ├── train_reward_model.py       # TRL RewardTrainer · regression head
│   ├── train_dpo.py                # TRL DPOTrainer · LoRA via PEFT
│   ├── train_ppo.py                # TRL PPOTrainer · advanced config
│   └── merge_lora.py               # merge_and_unload() · safetensors export
├── eval/
│   ├── run_eval.py                 # three-axis evaluation suite
│   ├── run_redteam.py              # adversarial query suite · PDF report (ART-005)
│   ├── score_reward_model.py       # reward model validation on held-out set
│   └── tasks/
│       ├── alignment_accuracy.yaml
│       ├── refusal_calibration.yaml
│       └── consistency.yaml
├── artefacts/
│   ├── generate_manifest.py        # artefact bundle manifest generator
│   ├── export_bundle.py            # ZIP export for regulator submission
│   └── schemas/
│       ├── preference_pair.json
│       ├── annotation_log.json
│       └── bundle_manifest.json
├── serve/
│   ├── ollama_load.sh              # GGUF convert + Ollama model load (MVP demo)
│   └── vllm_config.yaml            # vLLM server config (production path)
├── docker/
│   ├── argilla/
│   │   └── docker-compose.yml      # Argilla + Elasticsearch
│   └── mlflow/
│       └── docker-compose.yml      # MLflow tracking server
├── notebooks/
│   ├── 01_bootstrap_data.ipynb
│   ├── 02_reward_model.ipynb
│   ├── 03_dpo_finetune.ipynb
│   └── 04_demo_comparison.ipynb
└── README.md
```

---

## Quickstart

**Requirements:** Python 3.10+, Docker, ~30GB disk, 16GB RAM minimum.
GPU required for fine-tuning. No GPU required for annotation and artefact stages.

```bash
git clone https://github.com/raosiddharthp/AlignR
cd AlignR
pip install -r requirements.txt
```

**1. Start infrastructure**

```bash
docker compose -f docker/argilla/docker-compose.yml up -d
docker compose -f docker/mlflow/docker-compose.yml up -d
```

Argilla UI: `localhost:6900` · MLflow UI: `localhost:5000`

**2. Prepare preference data**

MVP demo (bootstrap):
```bash
python data/bootstrap/hh_rlhf_prep.py --output data/pref_dataset/ --n_pairs 5000
```

Production (domain expert annotation):
```bash
python data/argilla_setup.py --workspace alignr --domain insurance
# Annotators log in at localhost:6900
python data/export_preferences.py --workspace alignr --output data/pref_dataset/ --validate
```

**3. Train the reward model** *(Colab A100)*

```bash
python train/train_reward_model.py \
  --config configs/dpo_default.yaml \
  --dataset data/pref_dataset/ \
  --output ./reward_model/ \
  --mlflow-uri http://localhost:5000
```

Aborts if validation accuracy < 80% — this is a hard gate.

**4. Run DPO fine-tuning** *(Colab A100)*

```bash
python train/train_dpo.py \
  --config configs/dpo_default.yaml \
  --dataset data/pref_dataset/ \
  --reward-model ./reward_model/ \
  --output ./aligned_model_lora/ \
  --mlflow-uri http://localhost:5000
```

**5. Merge LoRA and export**

```bash
python train/merge_lora.py \
  --base meta-llama/Llama-3-8B-Instruct \
  --lora ./aligned_model_lora/ \
  --output ./aligned_model/
```

**6. Evaluate**

```bash
python eval/run_eval.py \
  --model ./aligned_model/ \
  --tasks alignment_accuracy,refusal_calibration,consistency \
  --output eval/results/ \
  --mlflow-uri http://localhost:5000

python eval/run_redteam.py \
  --model ./aligned_model/ \
  --suite eval/tasks/ \
  --output eval/redteam_report.pdf
```

**7. Generate artefact bundle**

```bash
python artefacts/generate_manifest.py \
  --model-version alignr-v1.0.0 \
  --mlflow-run-id <run_id> \
  --output artefacts/bundle/
```

**8. Load and demo locally (Ollama)**

```bash
bash serve/ollama_load.sh --model ./aligned_model/ --name alignr-v1.0.0
ollama run alignr-v1.0.0
```

---

## Configuration

| Scenario | Config | Key difference |
|---|---|---|
| First deployment · standard domain | `dpo_default.yaml` | β=0.1 · LoRA r=16 |
| Healthcare / high-stakes | `dpo_conservative.yaml` | β=0.2 · false-allow ≤1% |
| Compliance standards changing mid-cycle | `ppo_advanced.yaml` | Full PPO loop |
| Kaggle T4 16GB | `qlora_t4.yaml` | 4-bit QLoRA |

---

## Compliance artefacts

| ID | Artefact | Stage | Format |
|---|---|---|---|
| ART-001 | Annotation log | Preference collection | JSONL |
| ART-002 | Reward model weights + card | Reward model training | safetensors |
| ART-003 | Fine-tuning run record | DPO / PPO training | MLflow run export |
| ART-004 | Aligned model weights + card | LoRA merge | safetensors |
| ART-005 | Red-team report | Red-team evaluation | PDF |
| ART-006 | CCO / DPO sign-off | Release gate | Signed PDF |

```bash
python artefacts/export_bundle.py \
  --manifest artefacts/bundle/manifest.json \
  --output artefacts/alignr-v1.0.0-bundle.zip
```

---

## Stack

| Component | Tool |
|---|---|
| Base model | Llama-3-8B-Instruct |
| Fine-tuning | TRL ≥0.8 |
| PEFT / LoRA | PEFT ≥0.10 |
| Annotation | Argilla ≥2.0 |
| Experiment tracking | MLflow ≥2.10 |
| Evaluation | lm-evaluation-harness |
| Local inference | Ollama ≥0.3 |
| Production inference | vLLM ≥0.4 |

---

*Siddharth Rao · MLE Portfolio · 2026*
