"""
training_pipeline/train.py
==========================
Main training entrypoint using Red Hat's Training Hub.

Supports three training modes (set via config.training.algorithm):
  - "lora_sft"  → LoRA fine-tuning via Unsloth (recommended, fast, memory-efficient)
  - "sft"       → Full supervised fine-tuning via InstructLab-Training
  - "osft"      → Orthogonal Subspace FT (continual learning, preserves base skills)

Usage:
    python training_pipeline/train.py --config config/pipeline_config.yaml
    python training_pipeline/train.py --config config/pipeline_config.yaml --algorithm lora_sft
    python training_pipeline/train.py --config config/pipeline_config.yaml --dry-run
"""

import argparse
import logging
import subprocess
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
    """
    Returns model path: local if configured, else HuggingFace ID.
    The model will be downloaded automatically if not cached.
    """
    student_cfg = config["student"]
    local_path = student_cfg.get("model_path")
    if local_path and Path(local_path).exists():
        logger.info(f"Using local model: {local_path}")
        return local_path
    model_id = student_cfg["model_id"]
    logger.info(f"Using HuggingFace model: {model_id}")
    return model_id


def resolve_data_path(config: dict) -> str:
    """Returns path to formatted training JSONL."""
    final_dir = Path(config["data"]["final_dir"])
    chat_file = final_dir / "train_chat.jsonl"
    raw_file = final_dir / "train.jsonl"

    if chat_file.exists():
        logger.info(f"Using chat-formatted data: {chat_file}")
        return str(chat_file)
    elif raw_file.exists():
        logger.warning(f"Chat-formatted data not found. Using raw: {raw_file}")
        logger.warning("For best results, run: python training_pipeline/format_for_training.py")
        return str(raw_file)
    else:
        raise FileNotFoundError(
            f"No training data found in {final_dir}. "
            "Run prepare_data.py, run_sdg.py, mix_datasets.py, and format_for_training.py first."
        )


def run_lora_sft(config: dict, model_path: str, data_path: str, dry_run: bool = False):
    """
    Runs LoRA fine-tuning via Training Hub (Unsloth backend).

    This is the recommended mode for the GB10.
    Uses 4-bit NF4 quantization by default for maximum speed.
    """
    try:
        from training_hub import lora_sft
    except ImportError:
        logger.error("training_hub not installed. Run: pip install training-hub[cuda]")
        sys.exit(1)

    t_cfg = config["training"]
    lora_cfg = t_cfg.get("lora", {})
    ckpt_dir = Path(t_cfg["checkpoint_dir"]) / "lora"

    logger.info("=" * 60)
    logger.info("Starting LoRA fine-tuning (Unsloth backend)")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Data:  {data_path}")
    logger.info(f"  Output: {ckpt_dir}")
    logger.info(f"  LoRA r={lora_cfg.get('r', 16)}, alpha={lora_cfg.get('alpha', 32)}")
    logger.info(f"  Epochs: {t_cfg.get('num_epochs', 3)}")
    logger.info(f"  LR: {t_cfg.get('learning_rate', 2e-4)}")
    logger.info(f"  4-bit: {lora_cfg.get('load_in_4bit', False)}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN – skipping actual training call")
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = lora_sft(
        # Core paths
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=str(ckpt_dir),

        # LoRA parameters
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        load_in_4bit=lora_cfg.get("load_in_4bit", False),

        # Training hyperparameters
        num_epochs=t_cfg.get("num_epochs", 3),
        learning_rate=t_cfg.get("learning_rate", 2e-4),
        micro_batch_size=t_cfg.get("micro_batch_size", 8),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        max_seq_len=t_cfg.get("max_seq_len", 2048),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        lr_scheduler=t_cfg.get("lr_scheduler", "cosine"),

        # Logging & checkpointing
        save_steps=t_cfg.get("save_steps", 200),
        eval_steps=t_cfg.get("eval_steps", 200),
        logging_steps=t_cfg.get("logging_steps", 10),
    )

    logger.info(f"Training complete: {result}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")

    # Merge LoRA adapters into the base model for easy deployment
    _merge_lora_adapters(model_path, str(ckpt_dir), config)


def run_sft(config: dict, model_path: str, data_path: str, dry_run: bool = False):
    """
    Runs full supervised fine-tuning via Training Hub (InstructLab backend).

    Use this when you have enough compute and want maximum task performance.
    On the GB10 with 128GB VRAM, this works fine for 7B models.
    """
    try:
        from training_hub import sft
    except ImportError:
        logger.error("training_hub not installed. Run: pip install training-hub[cuda]")
        sys.exit(1)

    t_cfg = config["training"]
    ckpt_dir = Path(t_cfg["checkpoint_dir"]) / "sft"

    logger.info("=" * 60)
    logger.info("Starting full SFT (InstructLab backend)")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Data:  {data_path}")
    logger.info(f"  Output: {ckpt_dir}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN – skipping actual training call")
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = sft(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=str(ckpt_dir),
        num_epochs=t_cfg.get("num_epochs", 3),
        learning_rate=t_cfg.get("learning_rate", 1e-5),
        max_seq_len=t_cfg.get("max_seq_len", 2048),
        save_steps=t_cfg.get("save_steps", 200),
        logging_steps=t_cfg.get("logging_steps", 10),
    )

    logger.info(f"Training complete: {result}")


def run_osft(config: dict, model_path: str, data_path: str, dry_run: bool = False):
    """
    Runs Orthogonal Subspace Fine-Tuning (OSFT) via Training Hub.

    OSFT teaches new capabilities while PRESERVING existing model behavior.
    Ideal for: adding Text-to-SQL skill without degrading general language ability.
    This is Red Hat's own continual learning algorithm (now in HuggingFace PEFT).
    """
    try:
        from training_hub import osft
    except ImportError:
        logger.error("training_hub not installed. Run: pip install training-hub[cuda]")
        sys.exit(1)

    t_cfg = config["training"]
    ckpt_dir = Path(t_cfg["checkpoint_dir"]) / "osft"

    logger.info("=" * 60)
    logger.info("Starting OSFT – Orthogonal Subspace Fine-Tuning")
    logger.info("  (Preserves existing model capabilities while learning SQL)")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Data:  {data_path}")
    logger.info(f"  Output: {ckpt_dir}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN – skipping actual training call")
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    result = osft(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=str(ckpt_dir),
        unfreeze_rank_ratio=0.25,
        effective_batch_size=16,
        max_tokens_per_gpu=t_cfg.get("max_seq_len", 2048),
        max_seq_len=t_cfg.get("max_seq_len", 2048),
        learning_rate=t_cfg.get("learning_rate", 5e-6),
        num_epochs=t_cfg.get("num_epochs", 3),
    )

    logger.info(f"Training complete: {result}")


def _merge_lora_adapters(base_model_path: str, adapter_path: str, config: dict):
    """
    Merges LoRA adapters into the base model weights.
    The merged model can be loaded without PEFT for faster inference.
    """
    logger.info("Merging LoRA adapters into base model...")

    final_dir = Path(config["data"]["final_dir"])
    merged_dir = Path(config["training"]["checkpoint_dir"]) / "final_merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    merge_script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model_path}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, "{adapter_path}")

print("Merging adapters...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("{merged_dir}", safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained("{base_model_path}")
tokenizer.save_pretrained("{merged_dir}")

print(f"Merged model saved to: {merged_dir}")
"""

    result = subprocess.run(
        [sys.executable, "-c", merge_script],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        logger.info(f"✅ Merged model saved: {merged_dir}")
    else:
        logger.warning(f"Adapter merge failed (non-fatal): {result.stderr[:500]}")
        logger.info("You can still use the LoRA adapter directly from the checkpoint dir")


ALGORITHM_RUNNERS = {
    "lora_sft": run_lora_sft,
    "sft": run_sft,
    "osft": run_osft,
}


def main():
    parser = argparse.ArgumentParser(description="Run Training Hub finetuning")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--algorithm", choices=list(ALGORITHM_RUNNERS.keys()), default=None,
                        help="Override training algorithm from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and data paths without actually training")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve algorithm
    algorithm = args.algorithm or config["training"]["algorithm"]
    if algorithm not in ALGORITHM_RUNNERS:
        logger.error(f"Unknown algorithm: {algorithm}. Choose from: {list(ALGORITHM_RUNNERS.keys())}")
        sys.exit(1)

    logger.info(f"Training algorithm: {algorithm}")

    # Resolve paths
    model_path = resolve_model_path(config)
    data_path = resolve_data_path(config)

    # Run formatting if not already done
    chat_path = Path(config["data"]["final_dir"]) / "train_chat.jsonl"
    if not chat_path.exists():
        logger.info("Chat-formatted data not found – running formatter...")
        from training_pipeline.format_for_training import format_dataset
        raw_path = Path(config["data"]["final_dir"]) / "train.jsonl"
        format_dataset(raw_path, chat_path)

    # Run training
    runner = ALGORITHM_RUNNERS[algorithm]
    runner(config, model_path, data_path, dry_run=args.dry_run)

    logger.info("\n✅ Training complete!")
    logger.info("   Next step: python evaluation/evaluate.py")


if __name__ == "__main__":
    main()
