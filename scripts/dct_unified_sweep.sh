#!/usr/bin/env bash
# 整併版：學出來的旋轉平面 ＋ 量化後的整數係數 ＋ **交付即參數**。探索性質。
#
# 要回答什麼
# ────────────────────────────────────────────────────────────────────
# 現行做法是兩段拼起來的：擾動在重疊加窗的 STFT 上設計出來，交付時再壓成
# JPEG。最佳化在一個空間裡找解，交付把解投影到另一個空間，**投影削掉一塊**。
# 這一塊在兩個域上都量到了：
#
#   相位＋增益（STFT）  等失真位移 0.7012 → 0.5561       −21%
#   學習平面（浮點 DCT） 0.1535／0.6571 → 0.1617／0.5679  兩軸皆被支配
#
# DCT-Shield 沒有這一段——它的參數直接就是 JPEG 的整數係數。本批把同一個選擇
# 套到我們的參數化上，於是 `runs/dct_phase_design/README.md` §3.1 那條事前寫
# 下的預測變成可判定的：**若那 21% 真的來自事後投影，本批不該付。**
#
# 判讀規則（跑之前就寫下）
# ────────────────────────────────────────────────────────────────────
#   U1  **主判準。** 等失真下的未淨化位移要**高於 `nd_plane_25_qd`**
#       （DISTS 0.1617／位移 0.5679／擋下 6-of-10，CLIP 代理）。達不到就代表
#       那 21% 不是事後投影造成的，「參數住在交付格式裡就會比較好」這條推理
#       要整個重寫。比較一律走 `matched_distortion_table.py` 內插，
#       **固定 theta 下比不同條件是錯的**，本專案已有三個結論這樣出錯過。
#   U2  仍然要贏同上界的隨機解（`dct_unified_rand`）。每一個新參數化都要有
#       這一格，FND-004 就是栽在沒有它。
#   U3  `delta_within_1` 這一欄決定**新穎性怎麼寫**：比例高就代表我們動的
#       幾乎全落在 DCT-Shield 的 eps=1 球裡，論文只能主張「約束不同」，
#       不可以寫成「動作不同」。固定配對版在帶內工作點是 0.927。
#   U4  `zero_coef_frac` 是可行集稀薄程度的直接讀數（旋轉零向量還是零向量）。
#       固定配對版的成對零比例是 0.7538，但那些配對只帶 0.79% 的能量——
#       **這一欄高不等於容量不夠**，要與 U1 一起讀，不可單獨下結論。
#
# 兩筆已知的代價，跑之前就要記著
# ────────────────────────────────────────────────────────────────────
#   1. **殘差是稀疏高振幅尖峰。** 學習平面在等 DISTS 下多付 3.4 dB PSNR 與
#      2.2 倍 L∞。DISTS 打平不代表失真打平，等 PSNR／L∞ 對齊會明顯落後。
#   2. **空間選擇性變粗。** 8×8 格點的徑向解析度只有 8 階，而 32×32 的
#      rfft2 有 17 階。空間選擇性正是本方法對 DCT-Shield 的主要構造差異。
#
# theta 怎麼取
# ────────────────────────────────────────────────────────────────────
# 由 `nd_plane` 家族外推：浮點版 theta 2.5 是 DISTS 0.1285、事後投影版是
# 0.1617，差 0.033。故三點取 1.8／2.2／2.5，預期落在 0.125／0.146／0.162
# 附近，夾住失真帶 0.1286–0.1447。**落在帶外的點不可用於等失真比較**，
# 而 `matched_distortion_table.py` 拒絕外插——三點是為了確保帶內有得內插。
#
# 用法：bash scripts/dct_unified_sweep.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_dct_unified
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

# 十張、1000 步、`latent_norm`，與 `nd_plane` 家族逐字相同——**唯一的變因是
# 旋轉作用在量化的哪一側、以及交付什麼**。
COMMON="--data data/omniedit150 --loss latent_norm --steps 1000 --dct-qd 0.85"

POINTS="
du_plane_18:--conditions~dct_unified~--radius~1.8
du_plane_22:--conditions~dct_unified~--radius~2.2
du_plane_25:--conditions~dct_unified~--radius~2.5
du_rand_22:--conditions~dct_unified_rand~--radius~2.2
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

# **別人的卡一律不碰。**
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  echo "      每卡 3 個實測會 OOM（docs/OPERATIONS.md）" >&2
  exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $COMMON $(echo "$extra" | tr '~' ' ') \
      --images $IMGS > "$OUT/$tag.log" 2>&1 &
  echo "[unified] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
echo "[unified] 送出 $i 個（$(date)）"
