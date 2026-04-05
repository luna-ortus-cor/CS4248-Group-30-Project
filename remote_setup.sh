#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
PYBIN="$VENV_DIR/bin/python"

# All downloads/cache stay in the project folder
PIP_TMP="$(pwd)/.pip-tmp"
mkdir -p "$PIP_TMP"
export HF_HOME="$(pwd)/.hf-cache"
mkdir -p "$HF_HOME"
export TMPDIR="$PIP_TMP"

echo "Starting setup at $(date)"
echo "Venv: $VENV_DIR"

# ── Load GCC runtime ──────────────────────────────────────────────────────────
# numpy/scipy C extensions need libgomp, libgfortran, libstdc++ at runtime.
# srun spawns a non-interactive shell so /etc/profile and ~/.bashrc are not
# sourced automatically — the module system must be initialised explicitly.
_load_gcc_module() {
  # Source module init if module command is not yet available
  if ! command -v module >/dev/null 2>&1; then
    for init in /usr/share/modules/init/bash \
                /usr/local/Modules/init/bash \
                /opt/modules/init/bash \
                /etc/profile.d/modules.sh; do
      if [ -f "$init" ]; then
        # shellcheck disable=SC1090
        source "$init" && break
      fi
    done
  fi

  if command -v module >/dev/null 2>&1; then
    # Try common gcc module names on NUS HPC / typical clusters
    for name in gcc gcc/12 gcc/11 GCCcore GCCcore/12.3.0 GCCcore/11.3.0; do
      if module load "$name" 2>/dev/null; then
        echo "Loaded module: $name"
        return 0
      fi
    done
    echo "Warning: no gcc module found — numpy may fail to import on some nodes."
  else
    echo "Warning: module system not available — trying without."
  fi
}
_load_gcc_module

# Save LD_LIBRARY_PATH so remote_run.sh can restore it exactly,
# regardless of whether the module system works on the compute node.
echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH:-}\"" > .env_gcc
echo "Saved LD_LIBRARY_PATH → .env_gcc"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

if [ ! -x "$PYBIN" ]; then
  echo "Virtualenv python missing at $PYBIN" >&2
  exit 1
fi

# Health-check pip — recreate venv if it is corrupted
if ! "$PYBIN" -m pip --version >/dev/null 2>&1; then
  echo "pip is broken in existing venv — recreating..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
  if [ ! -x "$PYBIN" ]; then
    echo "Failed to recreate venv at $PYBIN" >&2
    exit 1
  fi
  "$PYBIN" -m ensurepip --upgrade || { echo "ensurepip failed" >&2; exit 1; }
fi

"$PYBIN" -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch — wheel source depends on CPU architecture.
# x86_64: use the official cu128 wheel index (faster, pinned CUDA version).
# aarch64: cu128 wheels don't exist for ARM; use PyPI which ships CUDA-enabled builds.
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"
if [ "$ARCH" = "aarch64" ]; then
  echo "Installing torch for aarch64 from PyPI..."
  "$PYBIN" -m pip install --no-cache-dir torch torchvision
else
  echo "Installing torch for x86_64 with CUDA 12.8 wheel..."
  "$PYBIN" -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

echo "Installing requirements..."
"$PYBIN" -m pip install --no-cache-dir -r requirements.txt

# Force-reinstall numpy to ensure a clean platform-native build.
echo "Reinstalling numpy for current platform..."
"$PYBIN" -m pip install --no-cache-dir --force-reinstall "numpy<2"

rm -rf "$PIP_TMP"

# Verify numpy imports correctly with current LD_LIBRARY_PATH
echo "Verifying numpy..."
"$PYBIN" -c "import numpy as np; print('numpy:', np.__version__, '— OK')" || {
  echo "ERROR: numpy import failed even after reinstall." >&2
  exit 1
}

# Verify CUDA
if command -v nvidia-smi >/dev/null 2>&1; then
  set +e
  "$PYBIN" - <<'PY'
import sys, torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| available:", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 3)
PY
  trc=$?
  set -e
  if [ $trc -ne 0 ]; then
    echo "WARNING: torch cannot see CUDA."
    exit 1
  fi
fi

echo "Setup complete at $(date)"
