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
#               條件、無法解析）。兩個候選各跑一個便宜的點取失真水位與單步成本。
#               `ln_time` 是同設定的 latent_norm 計時點，供階段三換算步數。
#   二、ig_*    **兩個抽法都掃**，各三個半徑：預檢的效率差被失真水位混淆，
#               分不出來，所以在等失真上比而不是先二選一。
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
#   階段二  bash scripts/image_guidance_sweep.sh "<卡號>" "ig_d18 ig_d21 ig_d25 ig_n25 ig_n30 ig_n35"
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

# `ln_ct` 是判準 G3 的**等總機時**對照，步數必須由 `ln_time` 與 `zt_*` 兩個
# 計時點換算，缺了就拒絕啟動——猜一個數字等於放棄這個對照。
needs_ln=0
for t in ${ONLY:-ig_d25}; do
  case "$t" in ln_ct) needs_ln=1;; esac
done
if [ "$needs_ln" = 1 ] && [ -z "${LN_STEPS:-}" ]; then
  echo "錯誤：ln_ct 要先設 LN_STEPS。它是判準 G3 的**等總機時**對照，" >&2
  echo "      步數要由 ln_time 與 zt_* 的 total_seconds 換算。" >&2
  exit 2
fi

# tag:額外旗標（~ 代表空白）:影像數（0 = 全部十張）
#
# `z_t` 的兩個抽法**都掃**：兩張圖的預檢裡 diffuse_src 的效率高 12%，但它同時
# 落在較高的失真上（0.157 對 0.091），而效率本來就隨失真下降——那 12% 分不出
# 是抽法的貢獻還是失真水位的貢獻。半徑各自選過，讓兩族都跨過失真帶
# （0.1286–0.1447）：預檢在 r=2.5 上 diffuse_src 是 0.157、noise 是 0.091。
POINTS="
zt_diffuse:--loss~image_guidance~--ig-zt~diffuse_src~--steps~200~--radius~2.5:2
zt_noise:--loss~image_guidance~--ig-zt~noise~--steps~200~--radius~2.5:2
ln_time:--loss~latent_norm~--radius~2.5~--steps~200:2
ig_d18:--loss~image_guidance~--ig-zt~diffuse_src~--radius~1.8:0
ig_d21:--loss~image_guidance~--ig-zt~diffuse_src~--radius~2.1:0
ig_d25:--loss~image_guidance~--ig-zt~diffuse_src~--radius~2.5:0
ig_n25:--loss~image_guidance~--ig-zt~noise~--radius~2.5:0
ig_n30:--loss~image_guidance~--ig-zt~noise~--radius~3.0:0
ig_n35:--loss~image_guidance~--ig-zt~noise~--radius~3.5:0
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
  args=$(echo "$extra" | tr '~' ' ' | sed "s/LNS/${LN_STEPS:-1000}/")
  if [ "$nimg" = "0" ]; then imgs="$IMGS"; else imgs=$(echo $IMGS | cut -d' ' -f1-"$nimg"); fi
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $args \
      --images $imgs > "$OUT/$tag.log" 2>&1 &
  echo "[image_guidance] $tag dev=$dev pid=$!  $args"
done
echo "[image_guidance] 送出 $i 個（$(date)）"
