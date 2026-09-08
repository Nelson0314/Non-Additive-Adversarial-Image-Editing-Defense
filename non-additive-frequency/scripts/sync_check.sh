#!/usr/bin/env bash
# 本機與遠端的程式碼是否同步。**清單由 `git status` 產生，不靠人記得改了哪些檔。**
#
# 為什麼要有這一支：漏傳一個檔不會報錯，只會讓遠端跑的是舊版。已經踩過兩次——
# 一次是漏傳派工腳本，五個新臂在遠端一整晚一格都沒跑成；一次是漏傳
# `tests/test_deliver_jpeg.py`，遠端的全套測試上多一個假的失敗，
# 而那個失敗看起來像是新改動造成的。
#
# 只比對版控會收的東西（`src/`、`scripts/`、`tests/`、`docs/`）。`runs/` 的
# 結果檔方向相反（由遠端回收），不在這裡。
#
# 用法：
#     bash scripts/sync_check.sh            # 只報告
#     bash scripts/sync_check.sh --push     # 報告並把不同步的傳上去
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

HOST=nelson0314@server.basiclab.lab.nycu.edu.tw
PORT=10102
REMOTE=/nfs/home/nelson0314/WACV-s3
PUSH=0
[ "${1:-}" = "--push" ] && PUSH=1

# `git status --porcelain` 的第二欄是路徑；重新命名的列是 `old -> new`，
# 取最後一段。目錄項（未追蹤的新目錄）展開成底下的檔案。
#
# **它印的路徑相對於 repo 根，不是 cwd。** 這個 repo 的根在上一層
# （`non-additive-frequency/` 只是其中一個目錄），不剝掉那個前綴的話
# `[ -f "$e" ]` 會全部落空，腳本會安靜地回報「沒有改動」——**比漏傳更糟，
# 因為它看起來是通過的**。
PREFIX=$(git rev-parse --show-prefix)
mapfile -t ENTRIES < <(git status --porcelain -- src scripts tests docs CLAUDE.md \
  | sed 's/^...//' | sed 's/.* -> //' | sed "s|^${PREFIX}||")
FILES=()
for e in "${ENTRIES[@]}"; do
  if [ -d "$e" ]; then
    while IFS= read -r f; do FILES+=("$f"); done \
      < <(find "$e" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' \))
  elif [ -f "$e" ]; then
    FILES+=("$e")
  fi
done
if [ ${#FILES[@]} -eq 0 ]; then
  echo "沒有未提交的程式碼改動，不需要比對。"
  exit 0
fi
echo "由 git status 取得 ${#FILES[@]} 個檔"

LOCAL=$(mktemp); REMOTE_SUMS=$(mktemp)
md5sum "${FILES[@]}" | sed 's|\*||' | sort -k2 > "$LOCAL"

# 取遠端的 md5，**並驗證那份清單是完整的**。
#
# **遠端的 `md5sum` 一定要包成恆為 0**：新檔在第一次推之前遠端還不存在，
# `md5sum` 對缺檔回傳非零，於是 ssh 也回非零，這道守門會把它讀成「連不上」
# 而拒絕推——正是新檔最需要被推的那一次。缺檔要由下面的行數比對認出來
# （`No such file` 也算一行），不是由回傳碼。
#
# 這道守門不是可有可無的：ssh 連不上時 REMOTE_SUMS 只剩一行錯誤訊息，
# 於是 MISSING 空、join 比不出東西、BAD 空，腳本印出「全部同步」。
# **伺服器連不上會被回報成完全同步**——比漏傳更糟，因為它看起來是通過的。
# 實測發生過：遠端斷線十一小時，這支仍然說全部同步。
fetch_remote_sums() {
  if ! ssh -o BatchMode=yes -o ConnectTimeout=30 -p "$PORT" "$HOST"         "cd $REMOTE && { md5sum ${FILES[*]} 2>&1 || true; }" > "$REMOTE_SUMS" 2>/dev/null; then
    echo "✗ 取不到遠端的 md5（ssh 失敗）。**不判定同步狀態。**" >&2
    rm -f "$LOCAL" "$REMOTE_SUMS"
    exit 4
  fi
  sort -k2 -o "$REMOTE_SUMS" "$REMOTE_SUMS"
  local got
  got=$(grep -cE "^[0-9a-f]{32}[[:space:]]|No such file" "$REMOTE_SUMS" || true)
  if [ "$got" -ne "${#FILES[@]}" ]; then
    echo "✗ 遠端回應不完整：${#FILES[@]} 個檔只拿到 $got 行。**不判定同步狀態。**" >&2
    head -3 "$REMOTE_SUMS" >&2
    rm -f "$LOCAL" "$REMOTE_SUMS"
    exit 4
  fi
}
fetch_remote_sums

MISSING=$(grep -i "No such file" "$REMOTE_SUMS" \
          | sed "s/^md5sum: //;s/: No such file.*//" || true)
DIFFER=$(join -j 2 "$LOCAL" "$REMOTE_SUMS" 2>/dev/null \
         | awk '$2!=$3 {print $1}' || true)
BAD=$(printf '%s\n%s\n' "$MISSING" "$DIFFER" | grep -v '^$' | sort -u)

if [ -z "$BAD" ]; then
  echo "✓ 全部同步"
  rm -f "$LOCAL" "$REMOTE_SUMS"
  exit 0
fi
echo "✗ 不同步："
echo "$BAD" | sed 's/^/  /'
if [ "$PUSH" -eq 0 ]; then
  echo "（加 --push 傳上去）"
  rm -f "$LOCAL" "$REMOTE_SUMS"
  exit 1
fi
while IFS= read -r f; do
  scp -q -P "$PORT" "$f" "$HOST:$REMOTE/$f" || { echo "  傳送失敗：$f" >&2; exit 3; }
  echo "  已傳 $f"
done <<< "$BAD"
# 傳完必須重新比對。scp 成功不代表內容相同——目標路徑不存在時它會建檔，
# 路徑寫錯就會在遠端多出一個沒人讀的檔而不是報錯。
fetch_remote_sums
STILL=$(join -j 2 "$LOCAL" "$REMOTE_SUMS" 2>/dev/null | awk '$2!=$3 {print $1}' || true)
rm -f "$LOCAL" "$REMOTE_SUMS"
[ -n "$STILL" ] && { echo "✗ 傳完仍不同步：$STILL" >&2; exit 3; }
echo "✓ 傳完並驗證同步"
