#!/usr/bin/env bash
set -euo pipefail

# Configuration
REMOTE_USER="kenji7"
REMOTE_HOST="xlogin.comp.nus.edu.sg"
REMOTE_TARGET="CS4248/CS4248-Group-30-Project"

echo "Syncing to ${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_TARGET}"
rsync -azP \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '.env' \
  --exclude '.claude' \
  --exclude 'models' \
  --exclude 'results' \
  --exclude '.hf-cache' \
  --exclude '.pip-tmp' \
  --exclude '.env_gcc' \
  --exclude 'pipeline/rag_store' \
  --exclude 'pipeline/rag_store_memecap' \
  --exclude 'pipeline/rag_store_full' \
  --exclude 'output' \
  . "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_TARGET}"

echo "Sync complete."
