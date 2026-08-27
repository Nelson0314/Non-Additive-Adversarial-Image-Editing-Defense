#!/usr/bin/env bash
# DCT 域保長配對旋轉的強度掃描。設計與判準見 `runs/dct_phase_design/README.md`。
#
# 要回答什麼
# ────────────────────────────────────────────────────────────────────
# 現行的「量化交付」是在**像素域**做完 STFT 相位旋轉、得到連續值影像、才壓成
# JPEG——最佳化在一個空間裡找解，交付把解投影到另一個空間。代價已經量到：
# 等失真下未淨化位移由 0.7012 掉到 0.5561（−21%）、擋下由 11/13 掉到 3/13
# （`runs/ip2p_deliver_jpeg/README.md` 第二節）。
#
# `dct_rotate` 的可行集**就是**交付集，沒有這一次投影。這一批要看那 21%
# 在不在。這是一條可證偽的預測，不是修辭。
#
# 判讀規則（跑之前就寫下）
# ────────────────────────────────────────────────────────────────────
#   D1  等失真下的未淨化位移**沒有高於**同失真的量化交付點，第 3.1 節那條
#       預測就是錯的，這條路只是換一個等價的參數化，不必再往抗淨化走。
#       對照點已量過：`ours_ph_q`（DISTS 0.0928／位移 0.4657 淨增益）與
#       `ours_pg_q20`（0.1947／0.6273）。**比較一律走等失真內插**
#       （`scripts/matched_distortion_table.py`），固定 theta 或固定半徑下
#       比不同條件是錯的，先前有三個結論這樣出錯過。
#   D2  `zigzag` 配對（對照組）與 `transpose` 在等失真下**分不開**，則
#       「保長只有在兩軸價錢相同時才有感知意義」這句話沒有實證支持，
#       論文不可以那樣寫。這一格是必跑的，不是備案。
#   D3  三個 theta 的失真沒有跨過 0.0928–0.1947 這一段，等失真內插做不了，
#       要補點再談。天花板（`ceiling.csv`，十張、不最佳化）指出帶內工作點
#       在 theta 約 1.04–1.13，故取 0.8／1.1／1.5 夾住它。
#
# **這一批只跑防禦。** 抗淨化是 `phase_retention.py` 讀已存的防禦圖再跑一輪，
# 要等 D1／D3 通過才值得排（一個條件 × 10 張 × 5 個 JPEG 算子 ≈ 1.94 GPU-h）。
#
# 用法：bash scripts/dct_rotate_sweep.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_dct_rotate
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

# **交付品質固定 0.85**，與 `ours_ph_q` 的 `--deliver-jpeg 0.85` 對齊，
# 這樣兩者的差別只剩「旋轉在量化的哪一側」這一項。
COMMON="--data data/omniedit150 --loss latent_norm --steps 1000 --dct-qd 0.85"

# tag:旗標
POINTS="
dct_rot_t08:--conditions~dct_rotate~--radius~0.8~--dct-pairing~transpose
dct_rot_t11:--conditions~dct_rotate~--radius~1.1~--dct-pairing~transpose
dct_rot_t15:--conditions~dct_rotate~--radius~1.5~--dct-pairing~transpose
dct_rot_zz11:--conditions~dct_rotate~--radius~1.1~--dct-pairing~zigzag
dct_rot_rand11:--conditions~dct_rotate_rand~--radius~1.1~--dct-pairing~transpose
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

# **別人的卡一律不碰**，這一道是強制的。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
# 每卡最多兩個 process（`docs/OPERATIONS.md`）。實測疊到四個會整批 CUDA OOM，
# 而且是跑了十幾分鐘之後才掛，所以這一道要擋在派工前面。
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
      --out "$OUT/$tag" $(echo "$extra" | tr '~' ' ') \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[dct_rotate] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
wait
