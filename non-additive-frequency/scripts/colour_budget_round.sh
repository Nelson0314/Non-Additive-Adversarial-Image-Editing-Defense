#!/usr/bin/env bash
# 色彩重映射鎖在衣物上，把**失真預算推到「整件衣服換一個顏色」**。
#
# 為什麼是這一批
# ────────────────────────────────────────────────────────────────────
# 使用者的框架有兩個方法，這一支跑第二個：「衣服的顏色不是重點，所以衣服的
# 顏色可以整個改變也沒關係，在此前提下防禦的失真就可以做到很大」。
#
# 色彩族至今**從來沒有在那個預算下跑過**：
#
#   runs/ip2p_color_hunt          全圖、無載體、radius 0.10
#   runs/ip2p_patch_tint_colour   衣物、radius 0.30／0.60，但**用位移量判**
#                                 （判為 12–14%「不成立」），而位移量量不到
#                                 「還認不認得出是誰」
#   runs/ip2p_content_constraint  colorgrid 臂 radius 0.30，**十格全部死在
#                                 輸出目錄不存在**（2026-09-05，已修）
#
# `ColorGridParam.project` 對半徑**沒有上限**：`radius=1.0` 表示仿射係數可以
# 偏離單位矩陣 ±1，足以把紅色整個映射成綠色。那就是使用者說的預算。
#
# 為什麼值得花這些機時
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_color_capacity_purify` 量到高容量色彩網格在**十個算子上全部保留
# 93–116%**（灰階 97%——AdvCF 說那是色彩攻擊唯一的死穴）。對照本專案的其他
# 載體：補丁族 blur σ1.5 只剩 21%、jpeg60 44%；相位族 JPEG30 只有 11%。
#
# **抗淨化在這一族是構造性的**（同半徑隨機對照也一樣平，1.05–1.30 倍），
# 缺的是效果。而效果從來沒有在大預算下量過。
#
# 四個臂
# ────────────────────────────────────────────────────────────────────
#   r06   radius 0.60   既有掃描的上限，接得回 ip2p_patch_tint_colour
#   r12   radius 1.20   「換一個顏色」的量級
#   r20   radius 2.00   遠超過換色所需，看效果會不會續漲
#   r20_rand            同半徑、隨機、零最佳化。**不可省**——這一族已知
#                       抗淨化對隨機沒有優勢，效果那一欄同樣要有地板才能讀
#
# 用法：bash scripts/colour_budget_round.sh "<卡號…>" [smoke]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
# **權重在系統共用快取，不在 `$HF_HOME`。** 那個變數由使用者的 shell profile
# 設定，而 `ssh <host> "bash script"` 是**非互動 shell、不載入 profile**，
# 於是腳本從 ssh 送出時 diffusers 找不到權重，錯誤訊息是
# 「model is not cached locally」＋一串 Hub 連線失敗——看起來像網路問題，
# 實際上是環境沒繼承。從互動 session 送同一支腳本則正常，故這個坑只在
# 自動化派工時出現。
export HF_HUB_CACHE="${HF_HUB_CACHE:-/var/cache/huggingface/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/var/cache/huggingface/assets}"
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-1 2 3 4 5})
MODE="${2:-full}"
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [smoke]" >&2; exit 2; }
if [ ${#DEVS[@]} -gt 5 ]; then
  echo "提醒：給了 ${#DEVS[@]} 張卡，裁到前五張（${DEVS[*]:0:5}）"
  DEVS=("${DEVS[@]:0:5}")
fi

OUT=runs/ip2p_colour_budget
LOG=runs/execution_logs/colour_budget.log
mkdir -p "$OUT" runs/execution_logs

CAT_TRAIN=clothing
CATS_REPLAY="accessory background"

# 容量固定在 `runs/ip2p_color_hunt` 找到的最佳工作點（16×16，49 152 個參數）。
# 那一批量到容量是槓桿：8×8 → 16×16 讓位移由 0.4072 升到 0.5441。
# **這一批動的是半徑不是容量**，兩個一起動就分不出是誰的功勞。
GRID=16
LUMA=16

arm_flags() {  # $1=臂 → 印出 "COND RADIUS"
  case "$1" in
    r06)      echo "color_grid 0.60" ;;
    r12)      echo "color_grid 1.20" ;;
    r20)      echo "color_grid 2.00" ;;
    r20_rand) echo "color_grid_rand 2.00" ;;
    *) return 1 ;;
  esac
}

# 影像清單由 `free` 臂實際跑過的格推導（與其他派工腳本同一個作法）。
# **`-type d`**：派工把每一格的 stdout 存成 `<格名>.log` 就在旁邊，
# `ls -d ...task_*` 會把日誌檔也算成影像。
#
# **取前 N 張，不是全部十張。** 這一批問的是「半徑開大之後效果的趨勢」，
# 六張足以看出斜率，而十張 × 四臂 = 40 格在五個並行位上要 7 小時。
# 取的是排序後的前 N 張，故它是 `free` 臂那十張的**子集**，逐圖仍可對照。
NIMG="${NIMG:-6}"
IMGS=$(find runs/ip2p_face_defence -maxdepth 1 -type d -name 'clothing_plain_task_*' \
       2>/dev/null | sed 's|.*/clothing_plain_||' | sort | head -n "$NIMG")
[ -z "$IMGS" ] && { echo "錯誤：推導不出影像清單" >&2; exit 2; }

COMMON="--data data/omniedit150 --attack-prompts data/attack_prompts.yaml \
--subject-source face --loss image_guidance --ig-zt diffuse_src \
--patch-carrier clothes --carrier-refine 4 --carrier-erode 3 --carrier-feather 8 \
--color-grid $GRID --color-luma-bins $LUMA \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002"

if [ "$MODE" = "smoke" ]; then
  IMG=$(echo $IMGS | awk '{print $1}')
  read -r COND RAD <<< "$(arm_flags r20)"
  echo "=== [$(date '+%F %T')] 燒測 r20（20 步）===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "${DEVS[0]}" || exit 3
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${DEVS[0]}" "$PY" scripts/ip2p_run.py \
    --out "$OUT/_smoke" --images "$IMG" --attack-category "$CAT_TRAIN" \
    --conditions "$COND" --radius "$RAD" --steps 20 $COMMON 2>&1 \
    | tee -a "$LOG" | tail -20
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "燒測失敗（rc=$rc）" | tee -a "$LOG"; exit 4; }
  [ -f "$OUT/_smoke/results.csv" ] || {
    echo "燒測沒有產出 results.csv" | tee -a "$LOG"; exit 4; }
  echo "✓ 燒測通過" | tee -a "$LOG"
  exit 0
fi

ARMS="r06 r12 r20 r20_rand"
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
# **只數 python，不數外殼。** 每一格是 `bash -c "python ip2p_run.py …"`，
# 外殼的指令字串裡同樣含 `ip2p_run.py`，所以 `grep -c` 會把一格算成兩個，
# 並行度變成設定值的一半——**不會報錯，只會慢一倍**。用 `comm` 欄位把外殼
# 濾掉。
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] 色彩預算掃描：$(echo $ARMS | wc -w) 臂 × $(echo $IMGS | wc -w) 張，$SLOTS 個並行位 ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  read -r COND RAD <<< "$(arm_flags "$arm")"
  for img in $IMGS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 45; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    TRAIN_OUT="$OUT/${arm}_${CAT_TRAIN}_$img"
    echo "[colour] $arm r=$RAD $img dev=$dev" | tee -a "$LOG"
    # 訓練跑 clothing 並存權重，另外兩類重播同一張防禦圖——色彩族的損失同樣
    # 只吃空字串的文字嵌入，三類解的是同一個最佳化問題。
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
      set -e
      '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN --conditions $COND --radius $RAD \
        --steps 6000 --step-size 0.01 --save-weights $COMMON
      for cat in $CATS_REPLAY; do
        '$PY' scripts/ip2p_run.py --out '$OUT/${arm}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat --conditions $COND --radius $RAD \
          --steps 0 --resume-weights '$TRAIN_OUT' $COMMON
      done
    " > "$OUT/${arm}_$img.log" 2>&1 < /dev/null &
    sleep 4
  done
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== [$(date '+%F %T')] 收工，results $(find $OUT -name results.csv | wc -l) 份 ===" | tee -a "$LOG"

# 讀數在 CPU 上補算，不佔卡。
ENT=""
for arm in $ARMS; do
  for d in "$OUT"/${arm}_*/; do
    [ -d "$d" ] && [ -f "$d/results.csv" ] || continue
    ENT="$ENT --entry $(basename "$d")=$d"
  done
done
# shellcheck disable=SC2086
"$PY" scripts/identity_probe.py --out "$OUT/identity.csv" $ENT 2>&1 | tee -a "$LOG"
echo "=== [$(date '+%F %T')] 身分讀數寫入 $OUT/identity.csv ===" | tee -a "$LOG"
