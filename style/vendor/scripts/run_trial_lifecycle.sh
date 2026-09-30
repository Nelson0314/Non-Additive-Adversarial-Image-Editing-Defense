#!/usr/bin/env bash
# 暫時性嘗試的生命週期：trials/<名稱>/ 不入版控，結束時升格或刪除，本機與遠端副本一起處理。
#
# 用法（專案根，經 vendor/scripts/run_trial_lifecycle.sh 呼叫）：
#   bash vendor/scripts/run_trial_lifecycle.sh new <名稱>                    建立 trials/<名稱>/ 與 README.md、PROMOTED 樣板
#   bash vendor/scripts/run_trial_lifecycle.sh promote <名稱> [--local-only]  升格：PROMOTED 列出的目的檔皆已提交後，
#                                                            在 docs/TRIALS.md 記錄目的檔、SHA-256 與 commit，刪除 trial
#   bash vendor/scripts/run_trial_lifecycle.sh drop <名稱> [--local-only]     放棄：docs/TRIALS.md 已提交且該列各欄齊全後刪除 trial
#
# PROMOTED 每行一個相對專案根的目的檔，限 src/、configs/、results/、scripts/、tests/、docs/ 之下；
# 每個檔案須已追蹤且與 HEAD 相同。至少一行。
#
# 遠端副本以 TRIAL_REMOTE（ssh 目標，含選項，例如 "-p 10101 user@host"）與 TRIAL_REMOTE_ROOT（遠端的
# 專案根，絕對路徑）處理；兩者未設定時拒絕，除非明確給 --local-only。遠端先核對專案身份
# （pyproject.toml 的 [project] name 與本機相同）、trials 與目標都不是符號連結、解析後路徑位於
# trials/ 之下且目標存在，刪除後再確認已不存在；遠端完成才刪本機。
# 名稱只允許小寫英數字與底線，不含日期、流水號或順序詞。
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[ -f "$ROOT/pyproject.toml" ] || { echo "[FATAL] $ROOT 不是專案根" >&2; exit 2; }
[ $# -ge 2 ] || { echo "用法：run_trial_lifecycle.sh new|promote|drop <名稱> [--local-only]" >&2; exit 2; }
action=$1; name=$2; option=${3:-}
[[ "$name" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "[FATAL] 名稱只允許小寫英數字與底線：$name" >&2; exit 2; }
[ -z "$option" ] || [ "$option" = --local-only ] || { echo "[FATAL] 未知選項 $option" >&2; exit 2; }
dir="$ROOT/trials/$name"
ledger="$ROOT/docs/TRIALS.md"

project_name() { grep -m 1 '^name *= *"' "$1/pyproject.toml" | cut -d '"' -f 2; }

local_trial() {  # 本機 trial 的路徑檢查：trials 與目標都不是符號連結，且解析後位於 trials/ 之下
  [ -d "$ROOT/trials" ] && [ ! -L "$ROOT/trials" ] || { echo "[FATAL] $ROOT/trials 不存在或是符號連結" >&2; exit 1; }
  [ -d "$dir" ] || { echo "[FATAL] 不存在：$dir" >&2; exit 1; }
  # 符號連結以 -L 判定；Windows 的 junction 不一定被 -L 認出，以解析後路徑判定，兩者同一訊息。
  { [ ! -L "$dir" ] && [ "$(cd "$dir" && pwd -P)" = "$(cd "$ROOT/trials" && pwd -P)/$name" ]; } \
    || { echo "[FATAL] $dir 是符號連結（或連結點），解析後不在 trials/ 之下" >&2; exit 1; }
}

REMOTE_SCRIPT='set -euo pipefail
root=$1; name=$2; expected=$3
cd "$root" || { echo "[FATAL] 遠端專案根不存在：$root" >&2; exit 3; }
[ -f pyproject.toml ] || { echo "[FATAL] 遠端 $root 不是專案根" >&2; exit 3; }
actual=$(grep -m 1 "^name *= *\"" pyproject.toml | cut -d "\"" -f 2)
[ "$actual" = "$expected" ] || { echo "[FATAL] 遠端專案為 $actual，預期 $expected" >&2; exit 3; }
[ -d trials ] && [ ! -L trials ] || { echo "[FATAL] 遠端 trials 不存在或是符號連結" >&2; exit 3; }
[ -d "trials/$name" ] || { echo "[FATAL] 遠端沒有 trials/$name" >&2; exit 3; }
{ [ ! -L "trials/$name" ] && [ "$(cd "trials/$name" && pwd -P)" = "$(cd trials && pwd -P)/$name" ]; } \
  || { echo "[FATAL] 遠端 trials/$name 是符號連結（或連結點），解析後不在 trials/ 之下" >&2; exit 3; }
rm -rf -- "trials/$name"
[ ! -e "trials/$name" ] || { echo "[FATAL] 遠端刪除後仍存在" >&2; exit 3; }
echo "$root/trials/$name"'

remove_remote() {
  [ "$option" = --local-only ] && return 0
  [ -n "${TRIAL_REMOTE:-}" ] && [ -n "${TRIAL_REMOTE_ROOT:-}" ] \
    || { echo "[FATAL] 未設定 TRIAL_REMOTE／TRIAL_REMOTE_ROOT；只處理本機時給 --local-only" >&2; exit 2; }
  [[ "$TRIAL_REMOTE_ROOT" =~ ^/[A-Za-z0-9._/-]+$ && "$TRIAL_REMOTE_ROOT" != *..* ]] \
    || { echo "[FATAL] TRIAL_REMOTE_ROOT 必須是不含 .. 的絕對路徑：$TRIAL_REMOTE_ROOT" >&2; exit 2; }
  local command removed
  command=$(printf 'bash -c %q trial %q %q %q' "$REMOTE_SCRIPT" "$TRIAL_REMOTE_ROOT" "$name" "$(project_name "$ROOT")")
  # shellcheck disable=SC2086
  removed=$(ssh $TRIAL_REMOTE "$command") || { echo "[FATAL] 遠端刪除失敗；本機副本保留" >&2; exit 1; }
  echo "[REMOVED] 遠端 $removed"
}

committed() {  # 已追蹤且與 HEAD 相同
  git -C "$ROOT" ls-files --error-unmatch -- "$1" >/dev/null 2>&1 \
    && git -C "$ROOT" diff --quiet HEAD -- "$1"
}

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
    cat > "$dir/PROMOTED" <<EOF
# 升格時逐行列出已搬入並提交的目的檔（相對專案根，限 src/、configs/、results/、scripts/、tests/、docs/）
EOF
    echo "[NEW] $dir"
    ;;
  promote)
    local_trial
    git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 || { echo "[FATAL] $ROOT 不在 git 工作目錄中" >&2; exit 1; }
    [ -f "$dir/PROMOTED" ] || { echo "[FATAL] 缺少 $dir/PROMOTED" >&2; exit 1; }
    targets=()
    while IFS= read -r line; do
      line=${line%%#*}; line=$(echo "$line" | xargs)
      [ -n "$line" ] && targets+=("$line")
    done < "$dir/PROMOTED"
    [ "${#targets[@]}" -gt 0 ] || { echo "[FATAL] PROMOTED 沒有列出任何目的檔" >&2; exit 1; }
    records=()
    for target in "${targets[@]}"; do
      [[ "$target" =~ ^(src|configs|results|scripts|tests|docs)/ && "$target" != *..* ]] \
        || { echo "[FATAL] 目的檔不在允許的目錄：$target" >&2; exit 1; }
      [ -f "$ROOT/$target" ] || { echo "[FATAL] 目的檔不存在：$target" >&2; exit 1; }
      committed "$target" || { echo "[FATAL] 目的檔未追蹤或與 HEAD 不同：$target" >&2; exit 1; }
      records+=("$target $(sha256sum "$ROOT/$target" | cut -d' ' -f1)")
    done
    commit=$(git -C "$ROOT" rev-parse HEAD)
    remove_remote
    [ -f "$ledger" ] || { echo "[FATAL] 缺少 $ledger" >&2; exit 1; }
    grep -q '^## 升格紀錄' "$ledger" \
      || printf '\n## 升格紀錄\n\n| 名稱 | 目的檔與 SHA-256 | commit |\n|---|---|---|\n' >> "$ledger"
    printf '| `%s` | %s | `%s` |\n' "$name" "$(printf '`%s`<br>' "${records[@]}")" "$commit" >> "$ledger"
    rm -rf -- "$dir"
    echo "[PROMOTED] $name（commit $commit；紀錄已加入 docs/TRIALS.md，需另行提交）"
    ;;
  drop)
    local_trial
    [ -f "$ledger" ] || { echo "[FATAL] 缺少 $ledger" >&2; exit 1; }
    committed "docs/TRIALS.md" || { echo "[FATAL] docs/TRIALS.md 有未提交的變更；先提交紀錄" >&2; exit 1; }
    row=$(grep -E "^\| *\`?$name\`?( |（|\|)" "$ledger" | head -n 1 || true)
    [ -n "$row" ] || { echo "[FATAL] 先在 docs/TRIALS.md 為 $name 寫一列（試了什麼、設定、關鍵數字、結論來源）" >&2; exit 1; }
    IFS='|' read -r _ c_name c_what c_setting c_numbers c_source _ <<< "$row"
    for cell in "$c_what" "$c_setting" "$c_numbers" "$c_source"; do
      [ -n "$(echo "$cell" | xargs)" ] || { echo "[FATAL] docs/TRIALS.md 中 $name 的欄位不可空白（未量測請寫「未量測」）" >&2; exit 1; }
    done
    [[ "$c_source" != *"trials/$name"* ]] || { echo "[FATAL] 結論來源指向即將刪除的 trials/$name" >&2; exit 1; }
    remove_remote
    rm -rf -- "$dir"
    echo "[DROPPED] $dir"
    ;;
  *)
    echo "[FATAL] 未知動作：$action" >&2; exit 2 ;;
esac
