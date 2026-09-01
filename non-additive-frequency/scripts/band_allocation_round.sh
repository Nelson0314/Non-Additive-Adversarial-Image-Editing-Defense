#!/usr/bin/env bash
# 把擾動的預算搬到「淨化之後還活著」的頻帶上。
#
# 為什麼
# ────────────────────────────────────────────────────────────────────
# 現行的頻率閘是 `jpeg_luma^0.25`，而 JPEG 亮度量化階**隨頻率遞增**（階大 =
# 人眼看不見 = 放行多），所以構造本身正在把預算往高頻推。高斯模糊在頻域乘的
# 是實正數 `|H_σ(f)| = exp(-2π²σ²f²)`，把那一帶的振幅整個拿掉。
#
# `runs/fixedpoint_blur` 算過 σ=1 的可贏上界（1.54 倍，最佳帶在 r 0.88–1.06），
# 但**從未涵蓋 σ=2**。用同一份 `runs/encoder_frequency_response/latent_norm.csv`
# 的 `move_per_dists` 乘上 σ=2 的存活振幅：
#
#     帶            敏感度      |H| σ1   |H| σ2    上界 σ1   上界 σ2
#     0.000–0.177      278     0.962    0.857        268      238
#     0.177–0.354      792     0.707    0.250        560      198
#     0.354–0.530    1 560     0.381    0.021        595       33
#     0.530–0.707    8 841     0.151    5.2e-4     1 337      4.6
#     0.707–0.884   46 938     0.044    3.7e-6     2 067     0.18
#     0.884–1.061  242 507     0.009    ~0         2 284     0.00
#
# σ=1 的最佳帶在高頻（複驗了原表），**σ=2 整個翻到 0.00–0.35**，比現行工作帶
# （0.53–0.88）高約兩個數量級。現行 `r_max = ∞` 的高通配置對 σ2 是最差的分配。
#
# 兩種搬法，各自有理由，故同批並列
# ────────────────────────────────────────────────────────────────────
#   硬上界   `--r-max`：通帶外直接歸零。乾淨、可解釋，但一刀切。
#   軟定價   `--survival-weight blur12`：閘乘上期望存活振幅
#            `(1 + Σ_σ exp(-2π²σ²f²)) / 3`，含 identity 那一項，所以只是
#            把預算「傾斜」而不是切掉。**編碼器對哪一帶敏感不寫進去**——
#            那由最佳化自己找，寫進去等於把六張圖量到的曲線焊進方法。
#
# 2 × 4 全因子：三個帶配置 ＋ 存活加權，各兩個半徑。半徑不對齊失真——
# 使用者已裁定這一輪失真不作錨點，照實報，並與 baseline 並列看圖。
#
# **損失一律 `latent_norm`**（只讀編碼器）。讀 UNet 的損失這一輪不用，太慢。
#
# 用法：bash scripts/band_allocation_round.sh "<四到五個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 4 ] && { echo "用法：$0 \"<至少四個卡號>\"" >&2; exit 2; }
# 卡是多人共用的。列印了卻不擋等於沒擋，所以這一道會 exit。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
grep -q -- "--survival-weight" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 --survival-weight，先 git pull" >&2; exit 2; }

OUT=runs/ip2p_band_allocation
mkdir -p "$OUT"
IMGS=$(cat runs/ip2p_fair_comparison/images10.txt | tr '\n' ' ')

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss latent_norm --steps 1000 --step-size 0.01 --save-weights --skip-existing"

# tag:額外旗標（`~` 代替空白，見其餘派工腳本的同一慣例）
POINTS="
hi_r25:--radius~2.5
hi_r40:--radius~4.0
mid_r25:--radius~2.5~--r-max~0.55
mid_r40:--radius~4.0~--r-max~0.55
lo_r25:--radius~2.5~--r-max~0.35
lo_r40:--radius~4.0~--r-max~0.35
surv_r25:--radius~2.5~--survival-weight~blur12
surv_r40:--radius~4.0~--survival-weight~blur12
"

# 每卡最多 2 個 process（`docs/OPERATIONS.md`）。疊到 3 個實測整批 OOM。
n_points=$(echo "$POINTS" | grep -c ':')
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
  echo "[band] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
