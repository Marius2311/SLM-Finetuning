"""
training_pipeline/train_direct.py
==================================
Direktes LoRA Training mit HuggingFace TRL + PEFT.
Umgeht Training Hub komplett – kompatibel mit NGC PyTorch Image (transformers 5.6, trl 1.2).

Usage:
    python3 training_pipeline/train_direct.py --config config/pipeline_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    t_cfg = config["training"]
    lora_cfg = t_cfg.get("lora", {})

    model_id = config["student"].get("model_path") or config["student"]["model_id"]
    data_path = Path(config["data"]["final_dir"]) / "train_chat.jsonl"
    if not data_path.exists():
        data_path = Path(config["data"]["final_dir"]) / "train.jsonl"
    ckpt_dir = Path(t_cfg["checkpoint_dir"]) / "lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model:  {model_id}")
    logger.info(f"Data:   {data_path}")
    logger.info(f"Output: {ckpt_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # LoRA config
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading dataset...")
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    # Format: apply chat template
    def format_example(ex):
        if "messages" in ex:
            # Already in chat format
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Raw format fallback
            text = f"Schema: {ex.get('schema', '')}\nQuestion: {ex.get('question', '')}\nSQL: {ex.get('sql', '')}"
        return {"text": text}

    formatted = [format_example(ex) for ex in examples]
    dataset = Dataset.from_list(formatted)
    logger.info(f"Dataset size: {len(dataset)}")

    # Training args
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
        dataloader_num_workers=0,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {ckpt_dir}...")
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
