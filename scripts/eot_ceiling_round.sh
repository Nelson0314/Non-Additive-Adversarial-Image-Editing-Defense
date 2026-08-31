#!/usr/bin/env bash
# 兩個問題一起問：把淨化放進訓練迴圈值多少，以及強度推到底之後天花板在哪。
#
# 為什麼是這兩個
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_band_allocation` 量到的處境是：未淨化的淨增益 0.577（佔 LPIPS
# 可達範圍 0.772 的 75%），但模糊 σ1 只剩 23%、σ2 剩 9%、裁切剩 13%。
# 那一批動的是頻帶配置，四個臂在 σ2 上的差距是 0.052 對 0.067——**在剩下的
# 一成裡挪動**。頻帶配置不是主要槓桿。
#
# 紀錄裡值錢的那一個是把淨化算子放進訓練迴圈：`ih_eot`（寬 EOT）在模糊 σ1
# 拿到 0.322，是本批四臂（0.131–0.159）的兩倍以上。**但那一組用的是讀 UNet
# 的 `image_guidance` 損失，慢。`eot_broad` ＋ `latent_norm` 這個組合從來沒
# 跑過**，而後者只過 VAE 編碼器。
#
# 第二個問題沒有人問過：**不管失真、把強度推到很誇張，σ2 與裁切那兩欄爬得
# 起來嗎？** 爬不起來的話，這個參數化在任何價錢下都贏不了那兩欄，那是一個
# 硬結論。`r80`／`r160_eot` 就是這個探針，失真會很難看，那是刻意的。
#
# 三個附帶問題各佔一格：
#   `floor04_eot`  頻譜加性下限。乘性參數化在平坦區動不了（|S| ≈ 0 乘什麼
#                  都接近零），加性下限是既有的逃生口，與 EOT 沒有一起跑過。
#   `surv_eot`     存活加權是上一批唯一沒有輸的那一項，看它在 EOT 上還留不留。
#
# 兩件對所有 EOT 格都做的修正
# ────────────────────────────────────────────────────────────────────
# 1. **模糊族擴到含 3.0。** 預設族是 {0.5,1,1.5,2}，而評測用的 σ=2 落在族的
#    **邊界**上，EOT 對邊界點最弱。擴到 {0.5,1,2,3} 讓它變成內點。
# 2. **裁切留在族裡。** `runs/ip2p_eot_geom_purify` 結掉的是**隨機化幾何**
#    那一支（每步抽放大倍率），不是 `eot_broad` 的裁切類別；後者從未單獨測過。
#
# 影像固定兩張（使用者指定，省時）：盆栽人與瑪利歐。
#
# 用法：bash scripts/eot_ceiling_round.sh "<五個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 5 ] && { echo "用法：$0 \"<五個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_eot_ceiling
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"
EOT="--purify-aware~eot_broad~--eot-sigmas~0.5~1.0~2.0~3.0"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss latent_norm --steps 1000 --step-size 0.01 --save-weights --skip-existing"

# 強度是 2×3 全因子（有無 EOT × 三個半徑），另加三格各問一件事。
POINTS="
r25:--radius~2.5
r40:--radius~4.0
r80:--radius~8.0
r25_eot:--radius~2.5~$EOT
r40_eot:--radius~4.0~$EOT
r80_eot:--radius~8.0~$EOT
r160_eot:--radius~16.0~$EOT
floor04_eot:--radius~4.0~--spectral-floor~0.04~$EOT
surv_eot:--radius~4.0~--survival-weight~blur12~$EOT
"

n_points=$(echo "$POINTS" | grep -c ':')
# 每卡最多 2 個 process（`docs/OPERATIONS.md`）。疊到 3 個實測整批 OOM。
if [ "$n_points" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n_points 個 process 需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
  echo "      只給了 ${#DEVS[@]} 張。每卡最多 2 個。" >&2
  exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[eotceil] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
