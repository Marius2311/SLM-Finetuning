"""
training_pipeline/train.py
==========================
Haupt-Trainingsskript für das Text-to-SQL Finetuning.

Nutzt direkt HuggingFace TRL + PEFT (LoRA) – getestet auf GB10 (ARM64/SM_121).

Getestete Versionskombination:
  - PyTorch 2.10 nv25.11 (NGC, SM_121)
  - peft 0.13.2 (--no-deps, kein torchao)
  - accelerate >=1.3.0
  - transformers 5.6.2 (NGC nativ)
  - trl 0.12.0 (NGC nativ)

Unterstützte Algorithmen (--algorithm):
  - lora_sft  → LoRA fine-tuning (Standard, empfohlen)

Usage:
    python3 training_pipeline/train.py --config config/pipeline_config.yaml
    python3 training_pipeline/train.py --config config/pipeline_config.yaml --algorithm lora_sft
    python3 training_pipeline/train.py --config config/pipeline_config.yaml --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("training_pipeline")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model_path(config: dict) -> str:
    student_cfg = config["student"]
    local_path = student_cfg.get("model_path")
    if local_path and Path(local_path).exists():
        logger.info(f"Nutze lokales Modell: {local_path}")
        return local_path
    model_id = student_cfg["model_id"]
    logger.info(f"Nutze HuggingFace Modell: {model_id}")
    return model_id


def generate_run_name(config: dict) -> str:
    """
    Generates a unique run name based on model, seed count, and epochs.
    Example: qwen0.5b_text2sql_v1_50seeds_3epochs
    """
    import re
    from datetime import datetime

    model_id = config["student"].get("model_path") or config["student"]["model_id"]
    # Extract short model name: "Qwen/Qwen2.5-0.5B-Instruct" → "qwen0.5b"
    model_short = model_id.split("/")[-1].lower()
    model_short = re.sub(r"-instruct$", "", model_short)
    model_short = re.sub(r"qwen2[.]5-", "qwen", model_short)
    model_short = re.sub(r"[^a-z0-9]", "", model_short)

    seeds = config["data"].get("sdg_seed_input_size") or config["data"].get("seed_sample_size", 0)
    epochs = config["training"].get("num_epochs", 3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    return f"{model_short}_text2sql_{seeds}seeds_{epochs}epochs_{timestamp}"


def resolve_data_path(config: dict) -> Path:
    final_dir = Path(config["data"]["final_dir"])
    chat_file = final_dir / "train_chat.jsonl"
    raw_file = final_dir / "train.jsonl"
    if chat_file.exists():
        logger.info(f"Nutze Chat-Format Daten: {chat_file}")
        return chat_file
    elif raw_file.exists():
        logger.warning(f"train_chat.jsonl nicht gefunden, nutze: {raw_file}")
        logger.warning("Für beste Ergebnisse: python3 training_pipeline/format_for_training.py")
        return raw_file
    else:
        raise FileNotFoundError(
            f"Keine Trainingsdaten in {final_dir}. "
            "Bitte zuerst prepare_data.py, run_sdg.py, mix_datasets.py und "
            "format_for_training.py ausführen."
        )


def run_lora_sft(config: dict, model_path: str, data_path: Path, dry_run: bool = False):
    """
    LoRA Fine-Tuning mit HuggingFace TRL + PEFT.
    Getestet auf GB10 (ARM64/SM_121) mit peft==0.13.2 + accelerate>=1.3.0.
    """
    t_cfg = config["training"]
    lora_cfg = t_cfg.get("lora", {})
    run_name = generate_run_name(config)
    ckpt_dir = Path(t_cfg["checkpoint_dir"]) / run_name
    logger.info(f"Run name: {run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LoRA Fine-Tuning (PEFT + TRL)")
    logger.info(f"  Modell:  {model_path}")
    logger.info(f"  Daten:   {data_path}")
    logger.info(f"  Output:  {ckpt_dir}")
    logger.info(f"  LoRA r={lora_cfg.get('r', 16)}, alpha={lora_cfg.get('alpha', 32)}")
    logger.info(f"  Epochen: {t_cfg.get('num_epochs', 3)}")
    logger.info(f"  LR:      {t_cfg.get('learning_rate', 2e-4)}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN – kein echtes Training")
        return

    import torch
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA verfügbar: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Tokenizer laden
    logger.info("Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Modell laden – direkt auf GPU schieben (kein device_map, kein accelerate nötig)
    logger.info("Lade Modell...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    ).to(torch.bfloat16).cuda()
    model.enable_input_require_grads()

    # LoRA konfigurieren
    logger.info("Konfiguriere LoRA...")
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset laden
    logger.info("Lade Dataset...")
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info(f"Geladene Beispiele: {len(examples)}")

    # Chat-Template anwenden
    def format_example(ex):
        if "messages" in ex:
            return {
                "text": tokenizer.apply_chat_template(
                    ex["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            }
        # Fallback für Raw-Format
        return {
            "text": (
                f"### Schema:\n{ex.get('schema', '')}\n\n"
                f"### Frage:\n{ex.get('question', '')}\n\n"
                f"### SQL:\n{ex.get('sql', '')}"
            )
        }

    formatted = [format_example(ex) for ex in examples]
    dataset = Dataset.from_list(formatted)
    logger.info(f"Dataset formatiert: {len(dataset)} Beispiele")

    # Training konfigurieren
    training_args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=t_cfg.get("num_epochs", 3),
        per_device_train_batch_size=t_cfg.get("micro_batch_size", 4),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=t_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=t_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        max_seq_length=t_cfg.get("max_seq_len", 2048),
        logging_steps=t_cfg.get("logging_steps", 10),
        save_steps=t_cfg.get("save_steps", 200),
        bf16=True,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        dataset_text_field="text",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starte Training...")
    trainer.train()

    logger.info(f"Speichere Modell nach {ckpt_dir}...")
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    logger.info("✅ Training abgeschlossen!")
    logger.info(f"   Checkpoints: {ckpt_dir}")
    logger.info("   Nächster Schritt: python3 evaluation/evaluate.py")


# Algorithmus-Registry – erweiterbar für spätere Algorithmen
ALGORITHM_RUNNERS = {
    "lora_sft": run_lora_sft,
}


def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL SLM Training")
    parser.add_argument("--config", default="config/pipeline_config.yaml",
                        help="Pfad zur Pipeline-Config")
    parser.add_argument("--algorithm", choices=list(ALGORITHM_RUNNERS.keys()),
                        default="lora_sft",
                        help="Trainingsalgorithmus")
    parser.add_argument("--dry-run", action="store_true",
                        help="Konfiguration prüfen ohne Training")
    args = parser.parse_args()

    config = load_config(args.config)

    logger.info(f"Algorithmus: {args.algorithm}")

    model_path = resolve_model_path(config)
    data_path = resolve_data_path(config)

    runner = ALGORITHM_RUNNERS[args.algorithm]
    runner(config, model_path, data_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
