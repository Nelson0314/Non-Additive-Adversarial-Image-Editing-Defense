#!/usr/bin/env bash
# 影像引導消除損失。**本專案第一個讀 UNet 的損失。**
#
# 依據
# ────────────────────────────────────────────────────────────────────
# IP2P 的取樣式（`pipeline_stable_diffusion_instruct_pix2pix.py:444-447`）裡，
# 影像引導是一個差：`s_I * [eps(z_t, c_I, null) - eps(z_t, 0, null)]`，而影像
# 無條件分支用的就是**零影像 latent**（同檔 :898）。
#
# 現行的 `latent_norm`（把 ‖E(x')‖ 壓向零）因此是「讓影像引導項消失」的
# **逐點版本**——它要求 E(x') 逐元素等於零張量。本批問的是函數版本會不會更
# 便宜：只要求 UNet 對兩者的**反應**相同，可行集是 UNet 的一個等位集，
# 比單點大得多。
#
# 三個階段，順序不可顛倒
# ────────────────────────────────────────────────────────────────────
#   一、zt      `z_t` 的抽法沒有預設（IP2P 由純噪聲起步，中間步的分布依賴
#               條件、無法解析）。兩個候選各跑一個便宜的點，選定之後才往下走。
#   二、ig_r*   選定的抽法上掃三個半徑，供等失真內插。
#   三、ln_ct   **同總機時**的 `latent_norm` 對照（判準 G3）。步數由
#               `LN_STEPS` 指定，**沒有預設**——它必須由階段一量到的單步成本
#               推出來，猜一個數字等於放棄這個對照。
#
# 事前判準（跑之前寫下，設計文件 §1.4）
# ────────────────────────────────────────────────────────────────────
#   G1  等 DISTS 下位移 >= latent_norm 的 1.10 倍。< 1.00 直接否決。
#   G2  擋下的型態是「重畫」而非「劣化」，逐張人眼判。
#   G3  贏過同總機時的 latent_norm，排除「多跑計算」。
#   G4  最後 10% 步的損失相對變化 < 1%，否則標為未收斂、不下結論。
#
# 用法：
#   階段一  bash scripts/image_guidance_sweep.sh "<卡號>" "zt_diffuse zt_noise"
#   階段二  IG_ZT=noise bash scripts/image_guidance_sweep.sh "<卡號>" "ig_r20 ig_r25 ig_r30"
#   階段三  LN_STEPS=350 bash scripts/image_guidance_sweep.sh "<卡號>" "ln_ct"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

OUT=runs/ip2p_image_guidance
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

# 主線的非加性設定，**唯一的變因是損失**。旗標與 `mainline_defense.sh` 的
# 相位臂逐字相同。
BASE="--data data/omniedit150 --conditions phase_gain --steps 1000 --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

# 派工前先確認遠端的驅動認得這些旗鈕（本地改完忘了同步已經發生過一次，
# 三個工作點在 argparse 被擋下、卡空轉半小時）。
for flag in --ig-zt --ig-t-min --ig-samples; do
  grep -q -- "$flag" scripts/ip2p_run.py || {
    echo "錯誤：scripts/ip2p_run.py 不認得 $flag，先同步本機的改動" >&2
    exit 2; }
done
grep -q "image_guidance" src/defense/image_guidance_loss.py 2>/dev/null || {
  echo "錯誤：找不到 src/defense/image_guidance_loss.py，先同步本機的改動" >&2
  exit 2; }

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

# 階段二與階段三各自需要一個**必須由量測決定**的值，缺了就拒絕啟動。
needs_zt=0; needs_ln=0
for t in ${ONLY:-ig_r25}; do
  case "$t" in ig_r*) needs_zt=1;; ln_ct) needs_ln=1;; esac
done
if [ "$needs_zt" = 1 ] && [ -z "${IG_ZT:-}" ]; then
  echo "錯誤：階段二要先設 IG_ZT=diffuse_src 或 IG_ZT=noise。" >&2
  echo "      這個選擇要由階段一（zt_*）的讀數決定，**不要猜一個預設**。" >&2
  exit 2
fi
if [ "$needs_ln" = 1 ] && [ -z "${LN_STEPS:-}" ]; then
  echo "錯誤：ln_ct 要先設 LN_STEPS。它是判準 G3 的**等總機時**對照，" >&2
  echo "      步數必須由 ig_r* 的 total_seconds 推出來。" >&2
  exit 2
fi

# tag:額外旗標（~ 代表空白）:影像數（0 = 全部十張）
POINTS="
zt_diffuse:--loss~image_guidance~--ig-zt~diffuse_src~--steps~200~--radius~2.5:2
zt_noise:--loss~image_guidance~--ig-zt~noise~--steps~200~--radius~2.5:2
ig_r20:--loss~image_guidance~--ig-zt~ZT~--radius~2.0:0
ig_r25:--loss~image_guidance~--ig-zt~ZT~--radius~2.5:0
ig_r30:--loss~image_guidance~--ig-zt~ZT~--radius~3.0:0
ln_ct:--loss~latent_norm~--radius~2.5~--steps~LNS:0
"

n=0
for p in $POINTS; do IFS=: read -r tag _ _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  exit 2
fi

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

i=0
for p in $POINTS; do
  IFS=: read -r tag extra nimg <<< "$p"
  selected "$tag" || continue
  args=$(echo "$extra" | tr '~' ' ' | sed "s/ZT/${IG_ZT:-noise}/; s/LNS/${LN_STEPS:-1000}/")
  if [ "$nimg" = "0" ]; then imgs="$IMGS"; else imgs=$(echo $IMGS | cut -d' ' -f1-"$nimg"); fi
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $args \
      --images $imgs > "$OUT/$tag.log" 2>&1 &
  echo "[image_guidance] $tag dev=$dev pid=$!  $args"
done
echo "[image_guidance] 送出 $i 個（$(date)）"
