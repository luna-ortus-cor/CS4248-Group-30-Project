#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# If not running inside a SLURM job, allocate a GPU node with srun
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Not in SLURM job. Submitting with srun..."
  SCRIPT="$(realpath "$0")"
  exec srun --unbuffered --label --gres="gpu:a100-80:1" --time=08:00:00 --mem=32G "$SCRIPT" "$@"
fi

PYBIN=".venv/bin/python"

if [ ! -x "$PYBIN" ]; then
  echo "Venv not found at $PYBIN. Run remote_setup.sh first." >&2
  exit 1
fi

LOG_DIR="output/logs"
mkdir -p "$LOG_DIR" output

echo "========================================================"
echo "Baseline benchmark run at $(date)"
echo "Host: $(hostname)"
echo "========================================================"
nvidia-smi -L || true

# Restore LD_LIBRARY_PATH saved by remote_setup.sh.
# This guarantees numpy/scipy can find libgomp/libgfortran on any compute node,
# even if the module system is unavailable or names differ between nodes.
if [ -f ".env_gcc" ]; then
  # shellcheck disable=SC1091
  source .env_gcc
  echo "Restored LD_LIBRARY_PATH from .env_gcc"
else
  echo "Warning: .env_gcc not found — run remote_setup.sh first."
fi

# ── Step 1: Build SigLIP RAG (skip if already built) ─────────────────────────
RAG_EMBEDDINGS="pipeline/rag_store/embeddings.npy"
RAG_METADATA="pipeline/rag_store/metadata.pkl"

if [ -f "$RAG_EMBEDDINGS" ] && [ -f "$RAG_METADATA" ]; then
  echo "RAG store already exists, skipping step 1."
else
  echo "Building SigLIP RAG knowledge base..."
  "$PYBIN" -u pipeline/build_siglip_rag.py 2>&1 | tee "$LOG_DIR/step1_build_rag.log"
fi

# ── Step 2: Run all models ────────────────────────────────────────────────────
# run_baseline.py loops over MODEL_REGISTRY automatically when no --model is given.
# Pass --model <key> here to run a single model instead.
echo "Running all models..."
"$PYBIN" -u pipeline/run_baseline.py "$@" 2>&1 | tee "$LOG_DIR/step2_inference.log"

# ── Step 3: Evaluate all predictions ─────────────────────────────────────────
echo ""
echo "========================================================"
echo "Step 3 — Evaluating all predictions"
echo "========================================================"

EVAL_LOG="$LOG_DIR/step3_evaluate.log"
"$PYBIN" -u pipeline/evaluate.py 2>&1 | tee "$EVAL_LOG"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Run complete at $(date)"
echo "Results saved to:"
echo "  output/preds_*.jsonl       — raw predictions per model"
echo "  output/eval_report.json    — metrics for all models"
echo "  $LOG_DIR/                  — per-step logs"
echo "========================================================"
