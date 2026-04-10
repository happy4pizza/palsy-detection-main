#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" src/data_pipeline/split_data.py "$@"
