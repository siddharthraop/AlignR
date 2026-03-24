# AlignR

**RLHF fine-tuning pipeline for regulated enterprise AI.** Takes an open-weight foundation model and produces a behaviourally aligned, enterprise-deployable model using preference data collected from domain experts — compliance officers, underwriters, legal counsel, claims adjusters. Every stage of the pipeline produces structured compliance artefacts that map directly to EU AI Act conformity assessment requirements.

> **Design doc →** [raosiddharthp.github.io/AlignR](https://raosiddharthp.github.io/AlignR)  
> **Pipeline simulator →** live in the design doc above

---

## The problem

General-purpose language models are trained to be broadly helpful. In regulated industries — insurance, banking, healthcare, legal services — broadly helpful is a liability. A model that tells a claimant their procedure "should be covered" is not being helpful. It is creating legal exposure.

The immediate fix is a system prompt. It works until it doesn't: context drift in long conversations, adversarial phrasing, edge cases the prompt didn't anticipate. Prompt engineering is a runtime intervention. It controls what the model is reminded of. It does not change what the model is.

The deeper problem is the EU AI Act. High-risk AI systems — those used in credit scoring, insurance, healthcare decision support, employment — require operators to produce documented conformity assessment evidence before deployment. That evidence must include training data provenance, a reproducible model version, documented human oversight, and proof of accuracy and robustness. A proprietary API call to GPT-4 or Claude satisfies none of these. The weights are inaccessible. The training chain is opaque. The vendor can push a model update that changes behaviour without notice.

The answer is weight-level behavioural alignment on an open-weight model, with a pipeline that produces the compliance artefacts as first-class outputs — not as an afterthought.

That is what AlignR builds.

---

## How it works

AlignR is a seven-stage pipeline. The core insight is that the pipeline run itself is the conformity assessment evidence package.

```
Base model → Preference collection → Reward model → Fine-tuning → Red-team → Sign-off → Deploy
     ↓               ↓                    ↓              ↓            ↓          ↓
  open-weight    annotation log      versioned       training      red-team    CCO sign-off
  Llama-3-8B     (ART-001)          weights          run record    report      document
                                    (ART-002)        (ART-003)     (ART-005)   (ART-006)
                                                  aligned model
                                                  (ART-004)
```

**Stage 1 — Base model selection.** Llama-3-8B-Instruct. Open weights, fits on a single A100 40GB for LoRA fine-tuning, strong instruction-following benchmark scores, Llama licence permits enterprise deployment under 700M MAU.

**Stage 2 — Preference data collection.** Domain experts — not ML engineers — use Argilla to review pairs of model responses and indicate which better satisfies institutional standards. Each annotation event is logged with annotator ID, timestamp, decision, confidence score, and free-text rationale. This log is ART-001 and is immutable after the annotation session closes.

**Stage 3 — Reward model training.** The preference dataset trains a reward model: a Llama-3-8B backbone with a regression head that assigns a scalar score to any (prompt, response) pair. Validation accuracy on held-out preference pairs must exceed 80% before training proceeds. The reward model weights are ART-002 — an inspectable artefact that encodes annotator judgment in a queryable form.

**Stage 4 — Fine-tuning.** DPO by default. The base model is fine-tuned against the preference signal using TRL's DPOTrainer with LoRA (r=16) via PEFT. This reduces trainable parameters from ~8B to ~20M, making fine-tuning feasible on a single A100. The full training config (hyperparameters, seed, dataset version hash) is logged to MLflow and forms ART-003. The aligned model weights, after LoRA merge, are ART-004.

**Stage 5 — Red-teaming.** The aligned model is evaluated against an adversarial query suite targeting known failure modes: coverage language leakage, refusal miscalibration, reward hacking, distributional edge cases. Results form ART-005. Any axis below threshold triggers a targeted retraining cycle — not a prompt-layer patch.

**Stage 6 — Compliance sign-off.** The CCO or DPO reviews the artefact bundle manifest and signs off on deployment. This signed document is ART-006 and is the release gate. No model is promoted to production without it.

**Stage 7 — Deployment.** The aligned model (merged, frozen) is served via Ollama for the MVP demo. vLLM is the documented production upgrade path. All inference runs within the private VPC. No weights or queries leave the enterprise boundary.

---

## Why DPO, not PPO

PPO-based RLHF requires reward model inference during the training hot path: sample a response, score it, compute a gradient, repeat. This makes the training loop complex, memory-intensive, and prone to instability.

DPO reformulates the RLHF objective as a supervised classification problem on preference pairs. Given a chosen response and a rejected response to the same prompt, DPO directly optimises the model to assign higher likelihood to the chosen response relative to the rejected one, with a KL penalty against the base model to prevent degeneration.

```
L_DPO = −log σ( β · log π_θ(y_w|x)/π_ref(y_w|x)  −  β · log π_θ(y_l|x)/π_ref(y_l|x) )
```

The result: equivalent or better alignment with a single-stage training loop, no online reward model inference, and a loss curve that is a direct compliance artefact. The reward model still exists — it is trained and versioned as ART-002 — but it is not in the training hot path.

PPO is supported as an advanced configuration for cases where the reward signal must be updated iteratively mid-cycle. DPO is the default because it is simpler, more stable, and produces cleaner audit trails.

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
│   ├── run_eval.py                 # lm-eval-harness runner
│   ├── run_redteam.py              # adversarial query suite
│   ├── score_reward_model.py       # reward model validation on held-out set
│   └── tasks/
│       ├── alignment_accuracy.yaml # held-out preference pair eval
│       ├── refusal_calibration.yaml
│       └── consistency.yaml        # paraphrase stability · tone consistency
├── artefacts/
│   ├── generate_manifest.py        # artefact bundle manifest generator
│   ├── export_bundle.py            # ZIP export for regulator submission
│   └── schemas/
│       ├── preference_pair.json    # JSON schema · preference pair
│       ├── annotation_log.json     # JSON schema · ART-001
│       └── bundle_manifest.json    # JSON schema · artefact bundle
├── serve/
│   ├── ollama_load.sh              # GGUF convert + Ollama model load (MVP demo)
│   └── vllm_config.yaml            # vLLM server config (production path)
├── docker/
│   ├── argilla/
│   │   └── docker-compose.yml      # Argilla + Elasticsearch
│   └── mlflow/
│       └── docker-compose.yml      # MLflow tracking server
├── docs/
│   └── index.html                  # design doc · raosiddharthp.github.io/AlignR
├── notebooks/
│   ├── 01_bootstrap_data.ipynb     # HH-RLHF prep walkthrough
│   ├── 02_reward_model.ipynb       # reward model training walkthrough
│   ├── 03_dpo_finetune.ipynb       # DPO fine-tuning walkthrough
│   └── 04_demo_comparison.ipynb    # base model vs. aligned model comparison
└── README.md
```

---

## Quickstart

**Requirements:** Python 3.10+, Docker, ~30GB disk (model checkpoints), 16GB RAM minimum. GPU required for fine-tuning (see compute section below). No GPU required for the annotation and artefact stages.

```bash
git clone https://github.com/raosiddharthp/AlignR
cd AlignR
pip install -r requirements.txt
```

**1. Start infrastructure**

```bash
# Argilla annotation platform
docker compose -f docker/argilla/docker-compose.yml up -d

# MLflow tracking server
docker compose -f docker/mlflow/docker-compose.yml up -d
```

Argilla UI available at `localhost:6900`. MLflow UI at `localhost:5000`.

**2. Prepare preference data**

For the MVP demo, bootstrap with Anthropic HH-RLHF:

```bash
python data/bootstrap/hh_rlhf_prep.py --output data/pref_dataset/ --n_pairs 5000
```

For production, configure Argilla and run an annotation campaign:

```bash
python data/argilla_setup.py --workspace alignr --domain insurance
# Annotators log in at localhost:6900 and review preference pairs
python data/export_preferences.py --workspace alignr --output data/pref_dataset/ --validate
```

**3. Train the reward model**

```bash
python train/train_reward_model.py \
  --config configs/dpo_default.yaml \
  --dataset data/pref_dataset/ \
  --output ./reward_model/ \
  --mlflow-uri http://localhost:5000
```

Reward model validation accuracy is logged to MLflow. Pipeline aborts if accuracy < 80% — this is a hard gate, not a warning.

**4. Run DPO fine-tuning**

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

Or run the notebook comparison: `notebooks/04_demo_comparison.ipynb`

---

## Configuration

All fine-tuning behaviour is controlled by YAML config. The recommended DPO default:

```yaml
# configs/dpo_default.yaml
strategy: dpo
base_model: meta-llama/Llama-3-8B-Instruct

lora:
  r: 16
  alpha: 32
  target_modules: [q_proj, v_proj, k_proj, o_proj]
  dropout: 0.05

dpo:
  beta: 0.1                  # KL penalty — lower = more drift from base allowed
  learning_rate: 1.0e-5
  batch_size: 8
  gradient_accumulation: 4
  epochs: 3
  warmup_ratio: 0.1
  dtype: bfloat16
  seed: 42

reward_model:
  epochs: 3
  learning_rate: 2.0e-5
  validation_threshold: 0.80  # hard gate — pipeline aborts below this

evaluation:
  alignment_accuracy_threshold: 0.85
  false_allow_threshold: 0.03
  paraphrase_stability_threshold: 0.85
```

**Config selection guide:**

| Scenario | Config | Key difference |
|---|---|---|
| First deployment · standard domain | `dpo_default.yaml` | β=0.1 · LoRA r=16 · recommended start |
| Healthcare / high-stakes | `dpo_conservative.yaml` | β=0.2 · false-allow threshold ≤1% |
| Compliance standards changing mid-cycle | `ppo_advanced.yaml` | Full PPO loop · iterative reward shaping |
| Kaggle T4 16GB (free tier) | `qlora_t4.yaml` | 4-bit quantised base · QLoRA adapters |

---

## Compute requirements

| Stage | Minimum | Recommended | Free tier option |
|---|---|---|---|
| Annotation (Argilla) | CPU · 4GB RAM | CPU · 8GB RAM | Local Docker |
| Reward model training | A100 40GB | A100 40GB | Google Colab |
| DPO fine-tuning (LoRA) | A100 40GB | A100 40GB | Google Colab |
| DPO fine-tuning (QLoRA) | T4 16GB | A100 40GB | Kaggle (30hr/week) |
| LoRA merge | CPU · 16GB RAM | CPU · 32GB RAM | Any machine |
| Evaluation | CPU or T4 | T4 16GB | Kaggle |
| Local inference (Ollama) | Apple M2 or RTX 3080 | M3 Pro or RTX 4090 | Consumer hardware |

End-to-end pipeline on Colab free tier (A100, session-limited): approximately 12–16 hours total compute across reward model and DPO stages. Colab sessions time out — checkpointing is enabled by default in both training scripts.

---

## Compliance artefacts

Every pipeline run produces six structured artefacts stored in MLflow. Together they constitute the EU AI Act conformity assessment evidence package for the deployed model version.

| ID | Artefact | Stage | Format |
|---|---|---|---|
| ART-001 | Annotation log | Preference collection | JSONL — annotator ID, timestamp, decision, rationale, IAA score |
| ART-002 | Reward model weights + card | Reward model training | safetensors + model card markdown |
| ART-003 | Fine-tuning run record | DPO / PPO training | MLflow run export — config YAML, loss curve, seed, dataset hash |
| ART-004 | Aligned model weights + card | LoRA merge | safetensors + model card markdown |
| ART-005 | Red-team report | Red-team evaluation | PDF — per-category pass/fail, failure analysis, reviewer sign |
| ART-006 | CCO / DPO sign-off | Release gate | Signed PDF — deployment scope, version ref, next review date |

```bash
# Export full artefact bundle as ZIP for regulator submission
python artefacts/export_bundle.py \
  --manifest artefacts/bundle/manifest.json \
  --output artefacts/alignr-v1.0.0-bundle.zip
```

The bundle manifest schema is in `artefacts/schemas/bundle_manifest.json`. All artefact paths include SHA-256 content hashes. Artefacts are immutable after the model version is released.

---

## Architecture decisions

Six Architecture Decision Records document the significant design choices in AlignR. Full context, alternatives considered, and consequences for each are in the [design doc](https://raosiddharthp.github.io/AlignR#adrs).

| ADR | Decision | Rationale |
|---|---|---|
| ADR-001 | Llama-3-8B-Instruct over proprietary API | Weight ownership is a prerequisite for EU AI Act conformity assessment |
| ADR-002 | DPO as default over PPO | Simpler training loop, more stable, cleaner audit trail. PPO supported as advanced config. |
| ADR-003 | LoRA r=16 over full fine-tuning | Full fine-tune requires 80GB+ VRAM. LoRA reduces trainable parameters to ~20M — feasible on A100 40GB. |
| ADR-004 | Argilla over Label Studio | Native preference ranking UI, built-in IAA, direct HuggingFace Dataset export. |
| ADR-005 | MLflow (self-hosted) over W&B | W&B sends run data to external servers. Incompatible with data governance model. |
| ADR-006 | Ollama for MVP demo, vLLM for production | vLLM requires dedicated GPU server. Ollama runs on consumer hardware with no infrastructure cost. |

---

## Known limitations

AlignR is honest about where it can fail. These are structural limitations of RLHF-based alignment, not bugs.

**Reward hacking.** The model can learn to produce outputs that score highly on the reward model without genuinely satisfying the alignment objective. Mitigated by the KL penalty (`beta` parameter), held-out reward model validation, and a red-team suite with reward-hacking probes.

**Annotator bias.** Preference data reflects who collected it. If annotators share blind spots, those blind spots become alignment targets. Mitigated by role diversity requirements (minimum three distinct roles per domain) and a Cohen's κ threshold of ≥0.7 per preference pair.

**Coverage gaps.** Edge cases and low-frequency query types are underrepresented in preference data. Mitigated by stratified sampling across query categories during annotation campaign design.

**Distributional shift.** As operating context evolves, the aligned model reflects the standards at annotation time, not current standards. Mitigated by a 6-month reassessment cycle and continuous output monitoring that triggers retraining when drift exceeds threshold.

**LoRA quality ceiling.** For highly domain-specific alignment objectives, a low-rank adapter may be insufficient. Documented upgrade path: increase LoRA rank or move to full fine-tuning on a 70B base model.

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| Base model | Llama-3-8B-Instruct | Meta · open-weight · Llama licence |
| Fine-tuning | TRL ≥0.8 | HuggingFace · DPO + PPO native |
| PEFT / LoRA | PEFT ≥0.10 | HuggingFace · LoRA + QLoRA |
| Annotation | Argilla ≥2.0 | Self-hosted · Docker Compose |
| Experiment tracking | MLflow ≥2.10 | Self-hosted · artefact registry |
| Evaluation | lm-evaluation-harness | EleutherAI · open source |
| Local inference | Ollama ≥0.3 | MVP demo · consumer hardware |
| Production inference | vLLM ≥0.4 | Documented upgrade path |
| Bootstrap dataset | Anthropic HH-RLHF | Public · HuggingFace Hub · MVP seed only |
| Containerisation | Docker ≥24 | Reproducible environments · image digests as artefacts |

Everything is free-tier or open source. The pipeline has no external data dependency at training or inference time after the initial base model download.

---

## Design doc

The full design documentation covers the problem statement (EU AI Act compliance gap), pipeline architecture with diagrams, component specifications, compliance artefact layer, deployment architecture, all six ADRs in MADR format, data schemas, MVP scope and build boundaries, failure modes, cost model, and glossary.

**[raosiddharthp.github.io/AlignR](https://raosiddharthp.github.io/AlignR)**

The design doc includes an interactive pipeline simulator demonstrating four regulated-industry scenarios (insurance DPO, banking PPO, healthcare safety-first, legal refusal calibration) with per-stage timing and live YAML output.

---

## Papers

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., NeurIPS 2022. Foundational RLHF paper.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., NeurIPS 2023. DPO algorithm.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., ICLR 2022.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., NeurIPS 2023.
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — HuggingFace, 2023.
- [EU Artificial Intelligence Act](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) — Regulation (EU) 2024/1689. High-risk AI system obligations.

---

## License

MIT. See [LICENSE](LICENSE).

---

*Siddharth Rao · MLE Portfolio · 2026*
