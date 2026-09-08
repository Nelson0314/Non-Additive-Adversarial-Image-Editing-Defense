#!/usr/bin/env bash
# 兩個新臂。**彼此無關，只是排在同一支裡送。**
#
# 一　`--loss facelock`：身分損失接在 VAE 的一次往返上，不經過 UNet
# ────────────────────────────────────────────────────────────────────
# 移植自 arXiv:2411.16832。判準（臉）與模組（走到哪裡）是兩個獨立的軸，
# 本專案此前只填了對角線：
#
#     latent_norm     只走 VAE      沒有臉
#     identity        走了 UNet     有臉      （實測 8/25，明顯較差）
#     **本臂**        **只走 VAE**  **有臉**  ← 從沒跑過
#
# 本專案特有、原論文沒有的結構：FaceLock 的擾動是全圖 L∞ 球，臉本身也被改；
# 這裡受保護主體在像素上凍結（`S ∩ M = ∅`），**臉完全沒有被動到**。於是
# `D(E(x'))` 的臉只能經由 VAE 的感受野被支撐內的內容影響——衣物上的圖樣要
# 隔著空間去改變重建出來的臉。**這一臂問的就是那件事做不做得到。**
# 做不到的話損失恆為常數、梯度為零，`stuck_at_init` 會滿。
#
# 成本：沒有擴散取樣的前向與反向，單張比補丁族低一到兩個數量級。
#
# 二　全圖色彩濾鏡：把支撐限制整個拿掉
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_colour_budget`／`colour_grid` 已經量到色彩族貼著自己的隨機對照
# （位移倍率 0.99–1.18）。診斷指出三個與 AdvCF（arXiv:2011.06690）的錯配，
# 其中一個是**可動區域**：AdvCF 對整張圖濾波、**包含被分類的物體本身**，
# 而本專案把受保護主體凍結，色彩只能碰衣物與背景。
#
# 這一臂把那個變因單獨拿掉：不給 `--subject-mask`、`--subject-source` 留預設，
# `apply_where` 因此是 `None`，色彩變換**作用在整張圖上**。
#
# **它違反本專案的威脅模型**（主體不再逐位元不動），故不是候選方案，
# 是一個歸因用的對照：色彩族的零效果裡，有多少是「不准碰主體」造成的。
# 出表時必須與其餘各臂分開列。
#
# 用法：bash scripts/facelock_colour_round.sh "<卡號…>" [臂…]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE="${HF_HUB_CACHE:-/var/cache/huggingface/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/var/cache/huggingface/assets}"
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-0 1 3 4 5})
# 預設仍是五張（`CLAUDE.md`：多人共用，佔滿會讓別人排不進來）。
# 超過五張**必須使用者明確授意**，故做成要另外給 `MAX_DEVS` 才放行，
# 不改預設值——改預設會讓之後每一批都靜默佔六張。
MAX_DEVS="${MAX_DEVS:-5}"
[ ${#DEVS[@]} -gt "$MAX_DEVS" ] && DEVS=("${DEVS[@]:0:$MAX_DEVS}")
ARMS="${2:-fl_patch col_full}"

OUT=runs/ip2p_facelock_colour
LOG=runs/execution_logs/facelock_colour.log
mkdir -p "$OUT" runs/execution_logs

CAT_TRAIN=clothing
CATS_REPLAY="accessory background"

NIMG="${NIMG:-6}"
IMGS=$(find runs/ip2p_face_defence -maxdepth 1 -type d -name 'clothing_plain_task_*' \
       2>/dev/null | sed 's|.*/clothing_plain_||' | sort | head -n "$NIMG")
[ -z "$IMGS" ] && { echo "錯誤：推導不出影像清單" >&2; exit 2; }

BASE="--data data/omniedit150 --attack-prompts data/attack_prompts.yaml \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002"

# 兩個臂的旗標**完全不同**，故各自展開，不共用一個 COMMON。
arm_cmd() {  # $1=臂 → 印出該臂專屬的旗標
  case "$1" in
    fl_patch)
      # 補丁載體 ＋ facelock 損失。載體那幾個旗標與 `patch_res_round.sh` 逐字相同，
      # 故它與帶限批、free 臂在**幾何上完全可比**，差別只在損失。
      echo "--subject-source face --loss facelock \
--conditions patch --patch-carrier clothes --carrier-refine 4 \
--carrier-erode 3 --carrier-feather 8 --patch-placement complement \
--radius 0.05 --steps 6000 --step-size 0.01 --save-weights" ;;
    col_full)
      # 全圖色彩濾鏡。**不給 --subject-mask、不給 --subject-source face**，
      # 於是 apply_where 是 None。半徑與容量取 `colour_budget` 的 r06／G=16，
      # 讓它與那一批只差「有沒有支撐限制」這一件事。
      echo "--loss image_guidance --ig-zt diffuse_src \
--conditions color_grid --color-grid 16 --color-luma-bins 16 \
--radius 0.60 --steps 6000 --step-size 0.01 --save-weights" ;;
    *) return 1 ;;
  esac
}

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] facelock ＋ 全圖色彩：$(echo $ARMS | wc -w) 臂 × $(echo $IMGS | wc -w) 張 ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  FLAGS=$(arm_cmd "$arm") || { echo "未知的臂 $arm" >&2; exit 2; }
  for img in $IMGS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 45; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    TRAIN_OUT="$OUT/${arm}_${CAT_TRAIN}_$img"
    echo "[$arm] $img dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
      set -e
      '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN $FLAGS $BASE
      for cat in $CATS_REPLAY; do
        '$PY' scripts/ip2p_run.py --out '$OUT/${arm}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat $FLAGS --steps 0 --resume-weights '$TRAIN_OUT' $BASE
      done
    " > "$OUT/${arm}_$img.log" 2>&1 < /dev/null &
    sleep 4
  done
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== [$(date '+%F %T')] 收工，results $(find $OUT -name results.csv | wc -l) 份 ===" | tee -a "$LOG"
