#!/usr/bin/env bash
# baseline 的執行環境：以本專案的 src 與 vendor 快照解析 Python 套件。
# 由 scripts/ 下的工具 source；呼叫端另可以 ENV_FILE 指定機器相關設定（例如 PY）。
BASELINE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -n "${ENV_FILE:-}" ]; then
  source "$ENV_FILE" || { echo "[FATAL] 無法載入 ENV_FILE=$ENV_FILE" >&2; return 1; }
fi
PY=${PY:-python}
export PYTHONPATH="$BASELINE_ROOT/src:$BASELINE_ROOT/vendor${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8
GPU_TOOLS="$BASELINE_ROOT/vendor/scripts"
