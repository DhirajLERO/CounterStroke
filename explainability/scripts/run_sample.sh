#!/usr/bin/env bash
# Run explainability for one sample (debug) on GPU cluster.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/run_sample.py "$@"
