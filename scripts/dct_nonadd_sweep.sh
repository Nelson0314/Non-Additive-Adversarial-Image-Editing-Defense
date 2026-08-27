#!/usr/bin/env bash
# DCT 域的**非加性**擾動：強度掃描。探索性質，不是要馬上當成果。
#
# 要回答什麼
# ────────────────────────────────────────────────────────────────────
# DCT-Shield 在量化後的整數係數上做**加性**擾動（`x' = JPEG_D(α + δ)`）。
# 本專案在同一個域上能做、而它做不到的是**非加性**的形式。
#
# 第一次嘗試（`dct_rotation.py`，逐**固定配對**的二維旋轉）在十張上失敗：
# 等失真下未淨化位移只有現行做法的 60%、擋下 0/10，而且 `zigzag` 與
# `transpose` 兩種配對規則**分不開**（差 0.9%）。那個 null result 指向一個
# 具體的懷疑——**卡住的不是「動哪些配對」，而是「配對」這個限制本身**。
#
# 本批把限制拿掉：旋轉的平面改成**學出來的**（`plane`），並配兩個對照——
# 平面共用（`shared_plane`，參數少兩個數量級）與非正交的乘性增益（`gain`）。
# 固定配對旋轉是 `plane` 的特例（把平面釘在兩個座標軸上），所以 `plane` 若
# 仍然打不過現行方法，**「DCT 域的保長重相位」這一族就可以整個收掉**，
# 而不只是「那個配對規則不好」。
#
# 三個與前一批的差別，每一個都是刻意的
# ────────────────────────────────────────────────────────────────────
#   1. **作用在未量化的浮點係數上、輸出浮點影像。** 量化交付改成獨立旗鈕
#      （`--deliver-jpeg`），與相位法的 `ours_ph_n` 對 `ours_ph_q` 同構。
#      理由是實測：量化把 66.2% 的配對歸零，但那些配對只帶有 **0.79%** 的
#      成對能量——**量化不是容量瓶頸**，綁進參數化只會讓變因混在一起。
#   2. **不做 4:2:0 色度次取樣。** 次取樣單獨造成的像素往返誤差最大到 0.81，
#      那會變成與參數無關的失真地板。拿掉之後 `theta = 0` 精確到 1.2e−15。
#   3. **YCbCr 的反矩陣由 `linalg.inv` 算出**，不用 JFIF 的常數（後者正逆各自
#      四捨五入，往返只互逆到 5.7e−7）。本檔不模擬 libjpeg，沒有理由付那筆。
#
# 判讀規則（跑之前就寫下）
# ────────────────────────────────────────────────────────────────────
#   P1  `plane` 在等失真下的未淨化位移**沒有超過 `ours_ph_n`**（DISTS 0.0497
#       ／位移 0.5913／擋下 7-of-10），則「DCT 域的保長重相位」整族收掉。
#       比較一律走 `matched_distortion_table.py` 內插，**固定 theta 下比不同
#       條件是錯的**，先前有三個結論這樣出錯過。
#   P2  **已由 CPU 天花板回答，不必花 GPU**：`shared_plane`（全域共用一個平面）
#       的天花板只到 0.0117，構造上進不了失真帶；`plane`（逐區塊、平面對齊
#       係數向量）是 0.2313。**逐區塊那 52 萬個參數確實買到了可達範圍**，
#       差 20 倍。故本批不列 `shared_plane`。
#   P3  `gain`（非正交）勝過 `plane`（正交），則**保長本身是負擔不是資產**，
#       非加性的價值不在正交性上，論文的敘事要跟著改。
#   P4  最佳化與同上界的 `dct_nonadd_rand` 在等失真下分不開 → 與 FND-004 同型，
#       是最佳化沒買到東西，不是參數化不行。**每個新參數化都要有這一格。**
#
# theta 的取法由 `scripts/dct_nonadditive_ceiling.py` 的天花板決定
# （純 CPU、不最佳化）。`matched_distortion_table.py` 拒絕外插，猜錯就是整批
# GPU 時間白花（`warp_triad.sh` 踩過一次）。
#
# 用法：bash scripts/dct_nonadd_sweep.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_dct_nonadd
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

COMMON="--data data/omniedit150 --loss latent_norm --steps 1000"

# tag:旗標
# theta 由天花板探針定；三點夾住 `ours_ph_n` 的 0.0497 與失真帶 0.1286–0.1447。
# theta 由 `runs/dct_nonadd/ceiling_aligned.csv` 定（十張、紋理閘、**對齊平面**）：
#   1.20 → 0.079   1.80 → 0.147   2.50 → 0.205   3.14 → 0.231
# 最佳化實測約落在天花板的 50–60%（前一批 `dct_rotate` 的比例），故這四點約
# 對應 DISTS 0.040–0.139，夾住 `ours_ph_n` 的 0.0497 與失真帶 0.1286–0.1447。
#
# `gain` 只取 1.1／1.6：θ = 2.2 時 `clip_fraction` 是 0.1058，**超過 N2 的
# 0.10 門檻**，那時「非加性」的輸出有一成像素是被值域夾出來的，不可比。
#
# **`shared_plane` 不列入**：它全域共用一個平面，無法對齊每個區塊，天花板
# 只到 0.0117（θ=π），**構造上就進不了失真帶**，沒有等失真的工作點可比。
POINTS="
nd_plane_12:--conditions~dct_nonadd~--dct-mode~plane~--radius~1.2
nd_plane_18:--conditions~dct_nonadd~--dct-mode~plane~--radius~1.8
nd_plane_25:--conditions~dct_nonadd~--dct-mode~plane~--radius~2.5
nd_plane_31:--conditions~dct_nonadd~--dct-mode~plane~--radius~3.1416
nd_gain_11:--conditions~dct_nonadd~--dct-mode~gain~--radius~1.1
nd_gain_16:--conditions~dct_nonadd~--dct-mode~gain~--radius~1.6
nd_rand_25:--conditions~dct_nonadd_rand~--dct-mode~plane~--radius~2.5
nd_plane_25_qd:--conditions~dct_nonadd~--dct-mode~plane~--radius~2.5~--deliver-jpeg~0.85
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }
for v in THA THB THC THG; do
  case "$POINTS" in *"$v"*)
    echo "錯誤：$v 還沒有填。先跑 scripts/dct_nonadditive_ceiling.py 決定 theta，" >&2
    echo "      **並且要用 --plane-init aligned**（預設值）。隨機平面在 63 維裡" >&2
    echo "      平均只抓得到 2/63 的區塊能量，量出來的不是天花板而是隨機解，" >&2
    echo "      實測低估 18.6 倍（0.0124 對 0.2313），會誤殺整個方向。" >&2
    exit 2 ;; esac
done

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
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
  echo "[dct-nonadd] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
wait
