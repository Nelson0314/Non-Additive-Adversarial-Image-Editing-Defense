#!/usr/bin/env bash
# 暫時性嘗試的生命週期：trials/<名稱>/ 不入版控，結束時升格或刪除。
#
# 用法（專案根，經 vendor/scripts/trial.sh 呼叫）：
#   bash vendor/scripts/trial.sh new <名稱>       建立 trials/<名稱>/ 與 README.md 樣板
#   bash vendor/scripts/trial.sh promote <名稱>   需要的內容已搬入 src/、configs/、results/ 並提交後，刪除 trial
#   bash vendor/scripts/trial.sh drop <名稱>      docs/TRIALS.md 已有該名稱的一列後，刪除本機與遠端的 trial
#
# drop 以 TRIAL_REMOTE（ssh 目標，含選項，例如 "-p 10101 user@host"）與 TRIAL_REMOTE_ROOT
# （遠端的專案根）刪除遠端 trials/<名稱>/；兩者未設定時拒絕，除非明確給 --local-only。
# 名稱只允許小寫英數字與底線，不含日期、流水號或順序詞。
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[ -f "$ROOT/pyproject.toml" ] || { echo "[FATAL] $ROOT 不是專案根" >&2; exit 2; }
[ $# -ge 2 ] || { echo "用法：trial.sh new|promote|drop <名稱> [--local-only]" >&2; exit 2; }
action=$1; name=$2; option=${3:-}
[[ "$name" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "[FATAL] 名稱只允許小寫英數字與底線：$name" >&2; exit 2; }
dir="$ROOT/trials/$name"
ledger="$ROOT/docs/TRIALS.md"

case "$action" in
  new)
    [ ! -e "$dir" ] || { echo "[FATAL] 已存在：$dir" >&2; exit 1; }
    mkdir -p "$dir"
    cat > "$dir/README.md" <<EOF
# $name

- 試了什麼：
- 設定（指令、參數、資料、種子）：
- 關鍵數字：
- 結論來源（CSV、圖、log 的位置）：
EOF
    echo "[NEW] $dir"
    ;;
  promote)
    [ -d "$dir" ] || { echo "[FATAL] 不存在：$dir" >&2; exit 1; }
    dirty=$(git -C "$ROOT" status --porcelain -- . ":(exclude)trials" 2>/dev/null) \
      || { echo "[FATAL] $ROOT 不在 git 工作目錄中" >&2; exit 1; }
    [ -z "$dirty" ] || { echo "[FATAL] 升格內容尚未提交：" >&2; echo "$dirty" >&2; exit 1; }
    rm -rf -- "$dir"
    echo "[PROMOTED] $name（commit $(git -C "$ROOT" rev-parse --short HEAD)）"
    ;;
  drop)
    [ -f "$ledger" ] || { echo "[FATAL] 缺少 $ledger" >&2; exit 1; }
    grep -Eq "^\| *\`?$name\`? *\|" "$ledger" \
      || { echo "[FATAL] 先在 docs/TRIALS.md 為 $name 寫一列（試了什麼、設定、關鍵數字、結論來源）" >&2; exit 1; }
    if [ "$option" != "--local-only" ]; then
      [ -n "${TRIAL_REMOTE:-}" ] && [ -n "${TRIAL_REMOTE_ROOT:-}" ] \
        || { echo "[FATAL] 未設定 TRIAL_REMOTE／TRIAL_REMOTE_ROOT；只刪本機時給 --local-only" >&2; exit 2; }
      # shellcheck disable=SC2086
      ssh $TRIAL_REMOTE "rm -rf -- '$TRIAL_REMOTE_ROOT/trials/$name'" \
        || { echo "[FATAL] 遠端刪除失敗" >&2; exit 1; }
      echo "[DROPPED] 遠端 $TRIAL_REMOTE_ROOT/trials/$name"
    fi
    rm -rf -- "$dir"
    echo "[DROPPED] $dir"
    ;;
  *)
    echo "[FATAL] 未知動作：$action" >&2; exit 2 ;;
esac
