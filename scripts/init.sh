#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="${PYTHON:-python3.11}"
SKIP_INSTALL=0

if [[ "${1:-}" == "--skip-install" ]]; then
  SKIP_INSTALL=1
fi

for directory in \
  data/raw/fit3d data/raw/human36m data/raw/other data/metadata \
  data/interim/static_frames data/processed data/splits \
  models/checkpoints models/calibrators \
  outputs/predictions outputs/metrics outputs/figures outputs/reports outputs/logs
do
  mkdir -p "$PROJECT_ROOT/$directory"
done

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
  "$VENV_DIR/bin/python" -m pip install -e "$PROJECT_ROOT"
  "$VENV_DIR/bin/python" -m ipykernel install --user --name physioload --display-name "Python (physioload)"
fi

echo "PhysioLoad environment is ready."

