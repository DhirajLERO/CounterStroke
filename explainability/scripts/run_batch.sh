#!/usr/bin/env bash
# Run explainability batch on GPU cluster from GracoImage/explainability/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec python -u scripts/run_batch.py "$@"
