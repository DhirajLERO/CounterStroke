#!/usr/bin/env bash
# Launch Streamlit without inotify file watching (cluster-safe).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
exec streamlit run analyse_final_population/app.py \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false \
  "$@"
