"""
evaluation/evaluate.py
=======================
Evaluates a finetuned Text-to-SQL model using Execution Accuracy (EX).

Execution Accuracy = fraction of test examples where the model's predicted
SQL produces the SAME result set as the ground-truth SQL when executed
against the actual database.

This is the gold standard metric for Text-to-SQL (used in Spider & BIRD benchmarks).

We also report:
  - Exact Match (EM): predicted SQL == gold SQL (normalized)
  - Per-complexity accuracy breakdown

Usage:
    python evaluation/evaluate.py --config config/pipeline_config.yaml --model-path ./data/final/checkpoints/final_merged
    python evaluation/evaluate.py --model-path ./data/final/checkpoints/lora --use-adapter
    python evaluation/evaluate.py --model-path ./data/final/checkpoints/lora --use-adapter --n-samples 100
"""

import argparse
import json
import logging
import re
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL normalization for exact-match comparison
# ---------------------------------------------------------------------------

def normalize_sql(sql: str) -> str:
    """
    Normalizes a SQL string for comparison:
    - Lowercase
    - Collapse whitespace
    - Remove trailing semicolon
    """
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";")
    return sql


# ---------------------------------------------------------------------------
# SQL execution for Execution Accuracy
# ---------------------------------------------------------------------------

def execute_sql_on_schema(sql: str, schema_ddl: str) -> tuple[bool, any]:
    """
    Executes a SQL query against an in-memory SQLite database
    created from the given schema DDL.

    Returns:
        (success: bool, result: any)
        On success: result is a sorted list of rows
        On failure: result is the error message
    """
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Execute the DDL statements (CREATE TABLE, INSERT INTO, etc.)
        # Split on ";" to handle multiple statements
        for stmt in schema_ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    cursor.execute(stmt)
                except sqlite3.Error:
                    pass  # Some DDL statements may fail (e.g. schema differences) – that's ok

        conn.commit()

        # Execute the query
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        # Sort rows for comparison (result set is unordered unless ORDER BY is used)
        result = sorted(str(r) for r in rows)
        return True, result

    except sqlite3.Error as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def execution_match(pred_sql: str, gold_sql: str, schema_ddl: str) -> bool:
    """
    Returns True if pred_sql and gold_sql produce the same result set
    when executed against the schema.
    """
    pred_ok, pred_result = execute_sql_on_schema(pred_sql, schema_ddl)
    gold_ok, gold_result = execute_sql_on_schema(gold_sql, schema_ddl)

    if not pred_ok or not gold_ok:
        return False

    return pred_result == gold_result


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, use_adapter: bool = False):
    """
    Loads the finetuned model for inference.
    Supports:
      - Merged model (no adapter)
      - LoRA adapter over base model
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if use_adapter:
        from peft import PeftModel
        # Find the base model ID from adapter config
        adapter_cfg_path = Path(model_path) / "adapter_config.json"
        if adapter_cfg_path.exists():
            with open(adapter_cfg_path) as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        else:
            base_model_id = "Qwen/Qwen2.5-7B-Instruct"
            logger.warning(f"adapter_config.json not found, assuming base: {base_model_id}")

        logger.info(f"Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


def predict_sql(
    model,
    tokenizer,
    question: str,
    schema: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Runs inference to predict SQL from a question + schema.
    Strips any <think>...</think> block from the output.
    """
    import torch

    system = "You are an expert SQL query writer. Given a natural language question and a database schema, write the correct SQL query. Output ONLY the SQL query."
    user_content = f"Database schema:\n{schema}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template (Qwen2.5 has a built-in one)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,      # Low temp for deterministic SQL
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Remove <think>...</think> block if present
    generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()

    return generated.strip()


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(
    model,
    tokenizer,
    test_examples: list[dict],
    n_samples: int | None = None,
) -> dict:
    """
    Runs evaluation on test examples.
    Returns a results dict with per-example and aggregate metrics.
    """
    if n_samples:
        import random
        test_examples = random.sample(test_examples, min(n_samples, len(test_examples)))

    logger.info(f"Evaluating on {len(test_examples)} examples...")

    results = []
    by_complexity = defaultdict(lambda: {"em": 0, "ex": 0, "total": 0})

    for i, ex in enumerate(test_examples):
        if i % 50 == 0:
            logger.info(f"  Progress: {i}/{len(test_examples)}")

        question = ex.get("question", "")
        schema = ex.get("schema", "")
        gold_sql = ex.get("sql", "")
        complexity = ex.get("complexity", "unknown")

        # Predict
        pred_sql = predict_sql(model, tokenizer, question, schema)

        # Exact Match
        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)

        # Execution Accuracy
        ex_match = execution_match(pred_sql, gold_sql, schema)

        results.append({
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "complexity": complexity,
            "exact_match": em,
            "execution_match": ex_match,
        })

        # Aggregate by complexity
        by_complexity[complexity]["total"] += 1
        if em:
            by_complexity[complexity]["em"] += 1
        if ex_match:
            by_complexity[complexity]["ex"] += 1

    # Overall metrics
    n = len(results)
    overall_em = sum(r["exact_match"] for r in results) / n
    overall_ex = sum(r["execution_match"] for r in results) / n

    return {
        "overall": {
            "n": n,
            "exact_match": round(overall_em, 4),
            "execution_accuracy": round(overall_ex, 4),
        },
        "by_complexity": {
            level: {
                "n": v["total"],
                "exact_match": round(v["em"] / v["total"], 4) if v["total"] else 0,
                "execution_accuracy": round(v["ex"] / v["total"], 4) if v["total"] else 0,
            }
            for level, v in by_complexity.items()
        },
        "examples": results,
    }


def print_results(results: dict):
    """Pretty-prints evaluation results."""
    overall = results["overall"]
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total examples:      {overall['n']}")
    logger.info(f"  Exact Match (EM):    {overall['exact_match']:.1%}")
    logger.info(f"  Execution Acc (EX):  {overall['execution_accuracy']:.1%}")
    logger.info("")
    logger.info("  Per-complexity breakdown:")

    order = ["basic SQL", "single join", "aggregation", "multiple joins",
             "subqueries", "window functions", "CTEs", "EXCEPT and INTERSECT"]

    for level in order + [k for k in results["by_complexity"] if k not in order]:
        if level not in results["by_complexity"]:
            continue
        row = results["by_complexity"][level]
        bar = "█" * int(row["execution_accuracy"] * 20)
        logger.info(f"  {level:30s}: EX={row['execution_accuracy']:.1%}  {bar}  (n={row['n']})")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned Text-to-SQL model")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model-path", required=True,
                        help="Path to merged model or LoRA adapter")
    parser.add_argument("--use-adapter", action="store_true",
                        help="Load as LoRA adapter (not merged model)")
    parser.add_argument("--test-file", default=None,
                        help="Override test data path")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Evaluate on a random subset")
    parser.add_argument("--output", default=None,
                        help="Save detailed results to this JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load test data
    test_path = Path(args.test_file) if args.test_file else \
                Path(config["data"]["final_dir"]) / "test.jsonl"

    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        return

    test_examples = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                # Handle both raw format and chat format
                if "messages" in ex:
                    # Extract from chat format
                    meta = ex.get("_meta", {})
                    # Skip chat-format – use raw test.jsonl instead
                    continue
                test_examples.append(ex)

    logger.info(f"Test examples: {len(test_examples)}")

    # Load model
    model, tokenizer = load_model(args.model_path, use_adapter=args.use_adapter)

    # Run evaluation
    results = evaluate(model, tokenizer, test_examples, n_samples=args.n_samples)

    # Print results
    print_results(results)

    # Save results
    # Eval-Ergebnisse in data/final/eval/<modellname>/results.json speichern
    model_name = Path(args.model_path).name
    eval_dir = Path(config["data"]["final_dir"]) / "eval" / model_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(eval_dir / "results.json")
    with open(output_path, "w") as f:
        # Don't serialize all examples by default (can be very large)
        summary = {k: v for k, v in results.items() if k != "examples"}
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
