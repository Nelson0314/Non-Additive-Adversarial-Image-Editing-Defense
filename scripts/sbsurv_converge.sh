#!/usr/bin/env bash
# 讓加性項與乘性項各自佔一個頻帶：乘性留在高頻買強度，加性搬到低頻買抗模糊。
#
# 為什麼
# ────────────────────────────────────────────────────────────────────
# 模糊 σ=2 是唯一沒有被任何路徑動過的一欄（換頻帶 1.28×、加強度在 radius 8
# 飽和、換損失後仍是 21% 可達）。殘差分析（`phase_shift_amp.png` 那一組）指出
# 原因：高斯模糊在頻域乘實正數 `exp(−2π²σ²f²)`，**構造上不可能改相位**，它抽掉
# 的是載體——相位在有能量的地方完好，看起來被轉的地方是因為能量已經沒了。
#
# 而「編碼器／UNet 對哪一帶敏感」與「哪一帶活得過模糊」方向相反：壓低 `--r-max`
# 會同時削掉乘性那一半的未淨化強度（實測掉四成以上）。分析上界說 σ2 的最佳帶
# 在 r < 0.35（編碼器 99×、UNet 190×），但實測只拿到 1.28×——缺口在於低頻的
# 容量被相位封頂（θ ≤ π）與格數（477 格只剩 40 格）夾死。
#
# **加性項不受這兩個限制**：它是加上去的，不是旋轉既有能量，而且它佔可用預算
# 的 67.6%（`runs/ip2p_residual_signature/allowed_budget_gini.csv`）。先前它與
# 乘性那一半共用同一個帶，所以沒辦法分開放。現在可以。
#
# 四個點
# ────────────────────────────────────────────────────────────────────
#   `sb_f35`      加性壓到 r ≤ 0.35，乘性不動（高通）
#   `sb_f55`      加性壓到 r ≤ 0.55，乘性不動
#   `sb_surv`     加性乘上期望存活振幅（軟版，不切邊），乘性不動
#   `sb_f35_surv` 兩者疊加
#
# 對照是既有的 `ig_f08_eot`（同損失、同下限 0.08、同 EOT，加性與乘性共用帶）。
#
# 收斂
# ────────────────────────────────────────────────────────────────────
# **步數不寫死。** `--steps` 只是上限，實際停止由 `--eval-every` ＋ `--patience`
# 的收斂判定決定，停在第幾步與原因記在 `stopped_at`／`stop_reason`。
# 先前那批的固定抽樣評估在 3000 步仍單調在降（最後 200 步還降 8%），也就是
# **結束訓練的是上限不是收斂**；本批把上限拉到 6000 並把 patience 放寬到 8 ×
# 400 步，讓煞車真的有機會踩下去。停在上限的一律照實回報為未收斂。
#
# 用法：bash scripts/split_band_round.sh "<兩個以上卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<兩個以上卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
grep -q -- "--floor-r-max" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 --floor-r-max，先 git pull" >&2; exit 2; }

OUT=runs/ip2p_sbsurv_long
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"
EOT="--purify-aware~eot_broad~--eot-sigmas~0.5~1.0~2.0~3.0"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 --spectral-floor 0.08 \
--steps 1600 --step-size 0.01 --resume-weights runs/ip2p_split_band/sb_surv \
--eval-every 400 --eval-draws 8 --patience 8 --min-delta 0.0002 \
--save-weights --skip-existing"

POINTS="
sb_surv_long:--floor-survival~blur12~$EOT
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
  echo "[splitband] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
