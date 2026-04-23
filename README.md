# Text-to-SQL SLM Finetuning Pipeline

End-to-end pipeline for finetuning a Small Language Model (SLM) on Text-to-SQL using:

- **SDG Hub** (Red Hat AI Innovation Team) – synthetic data generation
- **Training Hub** (Red Hat AI Innovation Team) – LoRA/SFT finetuning
- **NVIDIA GB10 (DGX Spark)** – 128 GB unified VRAM, runs via Docker over SSH

## Architecture Overview

```
gretelai/synthetic_text_to_sql (seed data)
        │
        ▼
┌─────────────────────────┐
│   SDG Pipeline          │  ← sdg_pipeline/
│   (SDG Hub + Teacher)   │
│   Teacher: Claude API   │    modular: swap to vLLM/Ollama
│         or local vLLM   │
└────────────┬────────────┘
             │  enriched JSONL
             ▼
┌─────────────────────────┐
│   Data Prep & Mixing    │  ← scripts/prepare_data.py
│   Format → chat JSONL  │
└────────────┬────────────┘
             │  training_data.jsonl
             ▼
┌─────────────────────────┐
│   Training Pipeline     │  ← training_pipeline/
│   (Training Hub LoRA)   │
│   Model: Qwen2.5-7B     │
│   Backend: Unsloth      │
└────────────┬────────────┘
             │  LoRA adapters
             ▼
┌─────────────────────────┐
│   Evaluation            │  ← evaluation/
│   Spider / BIRD bench   │
│   Execution Accuracy    │
└─────────────────────────┘
```

## Project Structure

```
text2sql_finetune/
├── config/
│   ├── pipeline_config.yaml       # Central config (model, paths, API)
│   └── teacher_backends.yaml     # Modular teacher model backends
├── sdg_pipeline/
│   ├── blocks/
│   │   └── sql_complexity_filter.py  # Custom SDG Hub block
│   ├── flows/
│   │   └── text2sql_enrichment.yaml  # SDG Hub flow definition
│   ├── prompts/
│   │   ├── complexity_upgrade.yaml   # Prompt: make SQL harder
│   │   ├── schema_variant.yaml       # Prompt: generate schema variants
│   │   └── reasoning_trace.yaml      # Prompt: add CoT reasoning
│   └── run_sdg.py                    # Main SDG entrypoint
├── training_pipeline/
│   ├── train.py                      # Main training entrypoint
│   └── format_for_training.py        # Data formatter (chat JSONL)
├── evaluation/
│   ├── evaluate.py                   # Execution accuracy evaluation
│   └── benchmark_spider.py           # Spider benchmark runner
├── scripts/
│   ├── prepare_data.py               # Download & prep seed data
│   └── mix_datasets.py               # Mix seed + synthetic data
├── docker/
│   ├── Dockerfile.sdg                # Container for SDG pipeline
│   ├── Dockerfile.training           # Container for training
│   └── docker-compose.yml            # Orchestrate both services
├── data/
│   ├── raw/                          # Downloaded seed dataset
│   ├── generated/                    # SDG Hub output
│   └── final/                        # Mixed & formatted for training
└── README.md
```

## Quick Start

### 1. Prerequisites (on your remote GB10 machine)

```bash
# Clone the repo
git clone https://github.com/Marius2311/SLM-Finetuning.git
cd SLM-Finetuning

# Edit: TEACHER_API_KEY in config/pipeline_config.yaml
```

### 2. Download and prepare seed data

```bash
docker compose -f docker/docker-compose.yml run --rm sdg \
  python scripts/prepare_data.py \
  --config config/pipeline_config.yaml
```

### 3. Run synthetic data generation

```bash
docker compose -f docker/docker-compose.yml run --rm sdg   python sdg_pipeline/run_sdg.py   --config config/pipeline_config.yaml
```

### 4. Mix and format datasets

```bash
docker compose -f docker/docker-compose.yml run --rm sdg \
  python scripts/mix_datasets.py \
  --config config/pipeline_config.yaml
```

### 5. Convert train data to chat format

```bash
# QWEN Instruct Model requires chat-like training data
docker compose -f docker/docker-compose.yml run --rm sdg \
  python training_pipeline/format_for_training.py \
  --config config/pipeline_config.yaml
```

### 6. Run finetuning

```bash
docker compose -f docker/docker-compose.yml build training

docker compose -f docker/docker-compose.yml run --rm training \
  python3 training_pipeline/train_direct.py \
  --config config/pipeline_config.yaml
```

### 7. Evaluate

```bash
docker compose -f docker/docker-compose.yml run --rm training \
  python3 evaluation/evaluate.py \
  --config config/pipeline_config.yaml \
  --model-path data/final/checkpoints/lora \
  --use-adapter \
  --n-samples 50
```

## Teacher Model Backends (Modular)

The SDG pipeline supports multiple teacher backends via `config/teacher_backends.yaml`:

| Backend              | Config key     | Use case                 |
| -------------------- | -------------- | ------------------------ |
| Anthropic Claude API | `anthropic`    | Default, highest quality |
| OpenAI API           | `openai`       | Alternative cloud API    |
| vLLM local           | `vllm_local`   | Local GPU, max privacy   |
| Ollama local         | `ollama_local` | Easiest local setup      |

Switch backends by setting `TEACHER_BACKEND` in your config.
