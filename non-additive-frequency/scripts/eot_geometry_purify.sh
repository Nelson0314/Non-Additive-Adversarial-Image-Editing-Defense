#!/usr/bin/env bash
# 補完 `runs/ip2p_eot_geometry` 缺的那一半：**抗淨化的淨增益**。
#
# 為什麼這一支不用訓練
# ────────────────────────────────────────────────────────────────────
# 那一批 13 張 × 3 個條件的防禦圖都還在。`phase_retention.py` 只讀已存的
# `*__def.png`，換淨化算子或加種子都不必重訓。這一批**只跑兩張**（使用者
# 指定的規模），兩張都在該批的名單裡。
#
# 為什麼這一格重要
# ────────────────────────────────────────────────────────────────────
# 隨機化幾何 EOT 是專案裡唯一被指名「可能救得了裁切」的機制，理由是
# `band_transfer` 與 `phase_drift_diagnosis` 都量到**裁切沒有破壞擾動、只是
# 把它整份搬走**（對搬過的餘弦 0.999、對原格點 0.005），也就是對位問題不是
# 能量問題。而 `make_eot_ops_transform` 裡的裁切是**固定**的中心 0.10，固定
# 的變換會被 co-adapt；`eot_geometry` 每步抽比例，評測用的 0.10 落在族內。
#
# 那一批**未淨化的代價已經量過**（`runs/ip2p_eot_geometry/README.md`：等失真
# 下位移掉 18–20%、擋下率掉超過一半），但它存在的理由從來沒跑過。要它有意義
# 就必須付得起那個代價，而付不付得起只有這一支答得出來。
#
# 空白地板重用
# ────────────────────────────────────────────────────────────────────
# 地板量的是**乾淨影像**過算子之後的編輯位移，與用哪個條件的防禦圖無關，
# 故沿用 `runs/ip2p_harden_purify` 的（同樣這兩張、同一組算子、同一顆種子）。
#
# **條件欄是 `phase` 不是 `phase_gain`**：那一批是純相位（`--gain-ratio 0`）。
#
# 用法：bash scripts/eot_geometry_purify.sh "<卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEV="${1:-}"
[ -z "$DEV" ] && { echo "用法：$0 \"<卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "$DEV" || exit 3

OUT=runs/ip2p_eot_geom_purify
GAL=runs/gallery_eot_geom
mkdir -p "$OUT" "$GAL"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"
PUR="identity jpeg90 jpeg75 jpeg50 jpeg30 blur1 blur2 crop_resize0.1 crop_resize0.15"

for f in floor_color floor_object; do
  [ -f "$OUT/$f.csv" ] || cp "runs/ip2p_harden_purify/$f.csv" "$OUT/$f.csv"
done

for t in g_r08 g_r18 g_rpi; do
  [ -f "$OUT/$t.csv" ] && { echo "$t 已有結果，跳過"; continue; }
  echo "$(date +%H:%M) $t 開始"
  CUDA_VISIBLE_DEVICES="$DEV" "$PY" scripts/phase_retention.py \
      --run "runs/ip2p_eot_geometry/$t" --images $IMGS --data data/omniedit150 \
      --attacker ip2p --seeds 1 --purifiers $PUR \
      --gallery "$GAL/$t" --out "$OUT/$t.csv" >> "$OUT/$t.log" 2>&1
  echo "$(date +%H:%M) $t 完成"
done
echo "$(date +%H:%M) 三個條件都跑完"
