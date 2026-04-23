#!/bin/bash
# =============================================================================
# scripts/deploy_vllm.sh
# =============================================================================
# Merged das LoRA-Modell mit dem Basismodell und startet einen vLLM Server.
#
# Usage:
#   ./scripts/deploy_vllm.sh <checkpoint_ordner>
#
# Beispiel:
#   ./scripts/deploy_vllm.sh qwen0.5b_text2sql_v1_50seeds_3epochs
# =============================================================================

set -e

CHECKPOINT_NAME=${1:-""}
if [ -z "$CHECKPOINT_NAME" ]; then
    echo "Fehler: Bitte Checkpoint-Ordner angeben."
    echo "Usage: ./scripts/deploy_vllm.sh <checkpoint_ordner>"
    echo ""
    echo "Verfügbare Checkpoints:"
    ls data/final/checkpoints/
    exit 1
fi

ADAPTER_PATH="data/final/checkpoints/${CHECKPOINT_NAME}"
MERGED_PATH="data/final/checkpoints/${CHECKPOINT_NAME}_merged"
PORT=8000

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Fehler: Checkpoint nicht gefunden: $ADAPTER_PATH"
    exit 1
fi

echo "============================================================"
echo "  Text-to-SQL Modell Deployment"
echo "  Adapter:  $ADAPTER_PATH"
echo "  Merged:   $MERGED_PATH"
echo "  Port:     $PORT"
echo "============================================================"

# ---------------------------------------------------------------------------
# Schritt 1: LoRA Adapter in Basismodell mergen (falls noch nicht geschehen)
# ---------------------------------------------------------------------------
if [ ! -d "$MERGED_PATH" ]; then
    echo ""
    echo "[1/2] Merge LoRA Adapter in Basismodell..."
    docker compose -f docker/docker-compose.yml run --rm training \
        python3 scripts/merge_adapter.py \
        --adapter-path "$ADAPTER_PATH" \
        --output-path "$MERGED_PATH" \
        --config config/pipeline_config.yaml
    echo "✓ Merge abgeschlossen: $MERGED_PATH"
else
    echo "[1/2] Merged Modell bereits vorhanden: $MERGED_PATH"
fi

# ---------------------------------------------------------------------------
# Schritt 2: vLLM Server starten
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] Starte vLLM Server auf Port $PORT..."
echo "      Zum Beenden: Ctrl+C"
echo ""

docker run --rm \
    --gpus all \
    --name text2sql_vllm_serve \
    -v "$(pwd)/$MERGED_PATH:/model" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -p ${PORT}:8000 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash -c "
        pip install -q vllm &&
        python3 -m vllm.entrypoints.openai.api_server \
            --model /model \
            --served-model-name text2sql \
            --host 0.0.0.0 \
            --port 8000 \
            --dtype bfloat16 \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.85
    "
