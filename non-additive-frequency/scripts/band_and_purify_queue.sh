#!/usr/bin/env bash
# 抗淨化優先的佇列：能立刻量抗淨化的先量，訓練排在後面。
#
# 為什麼是這個順序
# ────────────────────────────────────────────────────────────────────
# 使用者裁定**以抗淨化為主**。依此重排之後有兩件事跟著改：
#
#   1. **量化交付批的抗淨化排到最前面。** 那一批的防禦圖已經完整存在遠端
#      （54 格），抗淨化只讀圖不重訓，而它填的正是
#      `docs/DIRECTION.md` §6.1b 那張整片空著的十二算子表。此前它排在最後，
#      卡在約 29 小時的訓練後面——那是唯一「馬上就能拿到答案」的一段。
#
#   2. **每一批訓練完就立刻量它的抗淨化**，不再把三批的抗淨化併到最後。
#      理由是連線的可靠性：遠端已經連續斷線超過十小時，而「訓練完但沒量
#      抗淨化」的批次等於沒有回答任何問題（`jpeg_band_purify_round.sh`
#      的檔頭）。訓練與淨化交錯會讓卡在兩種工作之間切換，那個成本是可以
#      接受的；「跑完一整批卻拿不到結論」不行。
#
# 五段
# ────────────────────────────────────────────────────────────────────
#   1. 量化交付批的抗淨化   ← 不需訓練，直接量
#   2. 帶限批（燒測 → 正式）
#   3. 帶限批的抗淨化
#   4. 多邊形分割批（燒測 → 正式）
#   5. 多邊形分割批的抗淨化
#
# 帶限批排在多邊形分割之前，因為它是**專為抗低通設計**的那一個
# （S 的格點由 `runs/purifier_transfer/` 的模糊存活曲線反推：要讓 blur1.5
# 留下八成以上，結構必須粗於 16 像素，故 S ∈ {8,16,24,32}）。
# 多邊形分割主要是外觀與成本的答案，抗淨化是附帶的。
#
# 每一段送出前重新 `--assert` 卡：前一段剛結束，別人可能在空檔排進來。
#
# 用法：bash scripts/band_and_purify_queue.sh "<卡號…>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

CARDS="${1:-}"
[ -z "$CARDS" ] && { echo "用法：$0 \"<卡號…>\"" >&2; exit 2; }

LOG=runs/execution_logs/band_and_purify_queue.log
mkdir -p runs/execution_logs

running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^(python|python3)$/ && (/ip2p_run\.py/ || /purify_identity\.py/)' | wc -l
}
wait_idle() { while [ "$(running)" -gt 0 ]; do sleep 60; done; }

assert_cards() {
  bash scripts/free_cards.sh --assert "$CARDS" || {
    echo "=== [$(date '+%F %T')] 卡不是空的，佇列停在 $1 之前 ===" | tee -a "$LOG" >&2
    exit 3; }
}

# ── 一、量化交付批的抗淨化（不需訓練，最先做）────────────────
assert_cards "量化交付批的抗淨化"
echo "=== [$(date '+%F %T')] 量 deliver_jpeg 的抗淨化 ===" | tee -a "$LOG"
bash scripts/jpeg_band_purify_round.sh "$CARDS" deliver_jpeg || {
  echo "deliver_jpeg 抗淨化失敗，佇列中止。" | tee -a "$LOG" >&2; exit 6; }
wait_idle
echo "=== [$(date '+%F %T')] deliver_jpeg 抗淨化收工 ===" | tee -a "$LOG"
sleep 90

# ── 二、帶限批 ──────────────────────────────────────────────────
assert_cards "帶限批燒測"
echo "=== [$(date '+%F %T')] patch_res_round 燒測 ===" | tee -a "$LOG"
bash scripts/patch_res_round.sh "$CARDS" smoke || {
  echo "帶限批燒測失敗，佇列中止。" | tee -a "$LOG" >&2; exit 4; }
wait_idle
assert_cards "帶限批正式"
echo "=== [$(date '+%F %T')] patch_res_round 正式送出 ===" | tee -a "$LOG"
bash scripts/patch_res_round.sh "$CARDS" || {
  echo "帶限批送出失敗，佇列中止。" | tee -a "$LOG" >&2; exit 5; }
wait_idle
echo "=== [$(date '+%F %T')] patch_res_round 收工 ===" | tee -a "$LOG"
sleep 90

# ── 三、帶限批的抗淨化 ──────────────────────────────────────────
assert_cards "帶限批的抗淨化"
echo "=== [$(date '+%F %T')] 量 patch_res 的抗淨化 ===" | tee -a "$LOG"
bash scripts/jpeg_band_purify_round.sh "$CARDS" patch_res || {
  echo "patch_res 抗淨化失敗，佇列中止。" | tee -a "$LOG" >&2; exit 6; }
wait_idle
echo "=== [$(date '+%F %T')] patch_res 抗淨化收工 ===" | tee -a "$LOG"
sleep 90

# ── 四、多邊形分割批 ────────────────────────────────────────────
assert_cards "多邊形分割燒測"
echo "=== [$(date '+%F %T')] voronoi_round 燒測 ===" | tee -a "$LOG"
bash scripts/voronoi_round.sh "$CARDS" smoke || {
  echo "多邊形分割燒測失敗，佇列中止。" | tee -a "$LOG" >&2; exit 4; }
wait_idle
assert_cards "多邊形分割正式"
echo "=== [$(date '+%F %T')] voronoi_round 正式送出 ===" | tee -a "$LOG"
bash scripts/voronoi_round.sh "$CARDS" || {
  echo "多邊形分割送出失敗，佇列中止。" | tee -a "$LOG" >&2; exit 5; }
wait_idle
echo "=== [$(date '+%F %T')] voronoi_round 收工 ===" | tee -a "$LOG"
sleep 90

# ── 五、多邊形分割批的抗淨化 ────────────────────────────────────
assert_cards "多邊形分割的抗淨化"
echo "=== [$(date '+%F %T')] 量 voronoi 的抗淨化 ===" | tee -a "$LOG"
bash scripts/jpeg_band_purify_round.sh "$CARDS" voronoi || {
  echo "voronoi 抗淨化失敗，佇列中止。" | tee -a "$LOG" >&2; exit 6; }
wait_idle

echo "=== [$(date '+%F %T')] 佇列全部收工 ===" | tee -a "$LOG"
for d in runs/ip2p_deliver_jpeg_patch_purify runs/ip2p_patch_res          runs/ip2p_patch_res_purify runs/ip2p_voronoi runs/ip2p_voronoi_purify; do
  echo "$d: $(find "$d" -name results.csv 2>/dev/null | wc -l) 份 results.csv" | tee -a "$LOG"
done
