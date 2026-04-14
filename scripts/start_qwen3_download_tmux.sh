#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-qwen3-download}"
REPO_ID="${2:-Qwen/Qwen3-30B-A3B-Base}"
DEST_ROOT="${3:-$HOME/models}"
MODE="${4:-full}"

if [[ "$MODE" == "config" ]]; then
  CONFIG_FLAG="--config-only"
else
  CONFIG_FLAG=""
fi

tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

tmux new-session -d -s "$SESSION_NAME" \
  "unset ALL_PROXY; \
   unset all_proxy; \
   export HTTP_PROXY=http://public:114514@10.77.110.159:7890; \
   export HTTPS_PROXY=http://public:114514@10.77.110.159:7890; \
   cd $HOME/nano-ktrans; \
   python3 scripts/download_qwen3_moe.py --repo '$REPO_ID' --dest '$DEST_ROOT' $CONFIG_FLAG"

echo "Started tmux session: $SESSION_NAME"
echo "Inspect with: tmux capture-pane -pt ${SESSION_NAME}:0"
