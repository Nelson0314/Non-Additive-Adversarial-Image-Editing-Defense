#!/usr/bin/env bash
# 把色彩網格族跑到收斂：兩張影像 × 兩個損失，一格一卡。
#
# 為什麼是這一族、這個工作點
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_color/README.md`：網格族在兩個失真軸上都勝過同失真的隨機對照
# （DISTS 軸 1.50→1.11、LPIPS 軸 1.48→1.20），曲線族只有一個錨點可比、
# 倍率 1.09。同一份 README 也記著網格族**沒有收斂**——三個工作點在 4–6/6 張
# 影像上跑滿一千步、固定評估仍在往下走，所以那一批的每個網格讀數都是下界。
#
# 半徑取 0.10：它是網格族最強的一點（未淨化位移 0.4734），也是最不收斂的一點
# （六張全部 max_steps）。
#
# 為什麼兩個損失
# ────────────────────────────────────────────────────────────────────
#   latent_norm      ‖E(x_def)‖₂，只作用在 VAE 編碼器上（DCT-Shield §4.2 的目標）。
#                    `runs/ip2p_color` 整批用的就是它。
#   image_guidance   ‖ε(z_t, E_img(x'), ∅) − ε(z_t, 0, ∅)‖²，**唯一讀 UNet 的
#                    損失**；latent_norm 是它的逐點版本，可行集小得多。
#
# 兩者的可行集差很多，而色彩重映射的自由度只有 6144 個參數；
# 「收斂之後誰比較高」與「哪個損失配得上這個參數化」是同一個問題。
#
# 收斂判定用**與訓練目標無關的固定量測**（`--eval-every` ＋ `--patience`），
# 不看逐步損失——隨機目標的逐步損失本來就會抖。
#
# 用法：bash scripts/color_converge.sh "<卡號…>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\"" >&2; exit 2; }
# 卡是多人共用的。**這個檢查會 exit，不是印出來就算。**
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_color_converge
mkdir -p "$OUT"

RADIUS=0.10
# 一千步不夠。八千步 ＋ 每兩百步評估一次、連續八次沒有改善就停，
# 也就是最多再走一千六百步沒有進展才收手。
COMMON="--data data/omniedit150 --conditions color_grid --radius $RADIUS --steps 8000 --eval-every 200 --patience 8 --save-weights"

# `--ig-zt` 沒有預設值，且缺了它會在載入模型之後才拒絕（IP2P 由純噪聲起步，
# 中間步的 z_t 分布依賴條件、無法解析，兩個候選都是近似）。取 `diffuse_src`
# 是沿用專案既有的**全部** image_guidance 批次（`scripts/arch_sweep.sh`、
# `blur_band_round.sh`、`eot_blur_round.sh` 等九支派工腳本），換掉會讓新舊
# 批次不可比。
IG_EXTRA="--ig-zt diffuse_src"

# **每格一個獨立的輸出目錄**：每寫一列是整份重寫 CSV。
# 目錄名寫出「哪個損失、哪張圖」，不用流水號。
CELLS="latent_norm_plant:latent_norm:task_attr_mod_color_11699 \
latent_norm_mario:latent_norm:task_attr_mod_color_6205 \
image_guidance_plant:image_guidance:task_attr_mod_color_11699 \
image_guidance_mario:image_guidance:task_attr_mod_color_6205"

n=$(echo $CELLS | wc -w)
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 格需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

# 第二個參數可以只挑某些格跑（比對 tag 的前綴）。**踩過**：補跑
# image_guidance 時整份 CELLS 又送了一次，兩個 process 寫進同一個
# `latent_norm_*` 目錄——每寫一列是整份重寫 CSV，那會互相蓋掉。
ONLY="${2:-}"

i=0
for cell in $CELLS; do
  tag="${cell%%:*}"; rest="${cell#*:}"
  if [ -n "$ONLY" ]; then
    case "$tag" in $ONLY*) ;; *) continue ;; esac
  fi
  loss="${rest%%:*}"; img="${rest##*:}"
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  extra=""
  [ "$loss" = image_guidance ] && extra="$IG_EXTRA"
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --loss "$loss" --images "$img" $COMMON $extra \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[converge] $tag loss=$loss image=$img radius=$RADIUS dev=$dev"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"
