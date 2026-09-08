#!/usr/bin/env bash
# 無遮罩的 `image_guidance` 色彩網格**續跑到收斂**：兩個半徑 × 兩張影像。
#
# 為什麼是這一組
# `runs/ig_probe/` 量到色彩族真正管用的損失是 `image_guidance`（同失真下比
# 隨機多壓四成殘差），而 `latent_norm` 幾乎沒用（7–9%）；同一批也量到
# **加遮罩會讓殘差卡在 0.41**，不加遮罩的 r=0.20 是 0.015–0.023。所以這一族
# 若要留著，工作點就是「無遮罩 ＋ image_guidance」。
#
# `runs/ip2p_color_ig/` 的四格都是 `max_steps@8000`，讀數是下界。本批用
# `--resume-weights` 由那四格的存檔續跑，**不從零重來**——省一半機時，而且
# `resumed` 欄會逐列記下是不是真的載到了（載不到記 0，不靜默假裝續跑過）。
#
# 收斂判定用與訓練目標無關的固定量測：每 400 步以固定的 8 組 (t, ε) 評估，
# 連續 10 次（= 4000 步）沒有超過 min_delta 的改善才停。
#
# 一格一卡，16000 步約 2.2 小時。
#
# 用法：bash scripts/color_ig_converge_long.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_color_ig_converged
mkdir -p "$OUT"

COMMON="--data data/omniedit150 --conditions color_grid --loss image_guidance \
--ig-zt diffuse_src --steps 16000 --eval-every 400 --eval-draws 8 --patience 10 \
--min-delta 0.002 --save-weights"

# tag:半徑:影像:續跑來源。輸出目錄名帶進這一批的全部變因——每寫一列是整份
# 重寫 CSV，同名目錄先後跑兩組會把前一組整份蓋掉。
CELLS="ig_r005_plant:0.05:task_attr_mod_color_11699:runs/ip2p_color_ig/ig_r005_plant \
ig_r005_mario:0.05:task_attr_mod_color_6205:runs/ip2p_color_ig/ig_r005_mario \
ig_r020_plant:0.20:task_attr_mod_color_11699:runs/ip2p_color_ig/ig_r020_plant \
ig_r020_mario:0.20:task_attr_mod_color_6205:runs/ip2p_color_ig/ig_r020_mario"

ONLY="${2:-}"
n=$(echo $CELLS | wc -w)
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 格需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
for cell in $CELLS; do
  tag="${cell%%:*}"; rest="${cell#*:}"
  rad="${rest%%:*}"; rest="${rest#*:}"
  img="${rest%%:*}"; src="${rest##*:}"
  if [ -n "$ONLY" ]; then
    case "$tag" in $ONLY*) ;; *) continue ;; esac
  fi
  if [ ! -f "$src/${img}__color_grid__w.pt" ]; then
    echo "錯誤：$src 底下沒有 ${img} 的存檔，續跑會從零開始" >&2
    exit 2
  fi
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --radius "$rad" --images "$img" \
      --resume-weights "$src" $COMMON \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[color-ig] $tag radius=$rad image=$img dev=$dev 續跑自 $src"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"
