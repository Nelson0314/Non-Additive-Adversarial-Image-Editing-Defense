#!/usr/bin/env bash
# 同一個配方，往低失真掃：radius × 加性下限，外加一個把加性預算挪離主體的變體。
#
# 為什麼
# ────────────────────────────────────────────────────────────────────
# `ig_f08_eot`（UNet 影像引導損失 ＋ 加性下限 0.08 ＋ 寬 EOT）在六欄裡贏五欄，
# 但它的 DISTS 是 0.2289、殘差 RMS 0.100，人眼看是**把人臉打爛**。
#
# 失真主要由加性下限帶來：同半徑下 `ig_eot`（無下限）DISTS 0.1512／RMS 0.074，
# 開到 0.08 變成 0.2289／0.100，**下限一項就多付 51%**。所以下限是主要旋鈕，
# 半徑是次要旋鈕，兩個都往下掃。
#
# 第七格對準「臉」這件事，不是對準強度
# ────────────────────────────────────────────────────────────────────
# `docs/METHOD.md` 明寫**加法項的價目表不含紋理閘**——它只乘徑向帶通與頻率
# 定價，於是加性能量在畫面上是均勻噴的，臉也照噴。`--floor-gate complement`
# 把加性預算改花在「乘法那一半可達量最少」的區塊上，也就是通帶內幾乎沒有能量
# 的平坦區——那正是乘性參數化碰不到、而人眼也最不注意的地方。
#
# 既有量測支持它不貴：`runs/ip2p_residual_signature/` 記到 `complement` 在效果
# 上幾乎免費（位移 −0.4%、PSNR +0.28 dB）。**但它沒有和加性下限 0.08 或與
# UNet 損失一起跑過。**
#
# 收斂
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_ig_loss/ig_f08_eot/trace.csv` 的固定抽樣評估：0.145 → 0.0044
# （第 1000 步）→ 0.0023（第 3000 步）。**第 1000 步已走完 98.4% 的降幅，但
# 曲線到 3000 仍單調在降**，故沿用 3000 步，不縮。
#
# 影像兩張（使用者指定）：盆栽人與瑪利歐。
#
# 用法：bash scripts/ig_floor_lowdistortion.sh "<四個以上卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 4 ] && { echo "用法：$0 \"<四個以上卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_ig_lowdist
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"
EOT="--purify-aware~eot_broad~--eot-sigmas~0.5~1.0~2.0~3.0"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src \
--steps 3000 --step-size 0.01 \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002 \
--save-weights --skip-existing"

# 對照是既有的 `ig_f08_eot`（radius 2.5、下限 0.08、DISTS 0.2289），不重跑。
POINTS="
r25_f04:--radius~2.5~--spectral-floor~0.04~$EOT
r25_f02:--radius~2.5~--spectral-floor~0.02~$EOT
r18_f08:--radius~1.8~--spectral-floor~0.08~$EOT
r18_f04:--radius~1.8~--spectral-floor~0.04~$EOT
r12_f08:--radius~1.2~--spectral-floor~0.08~$EOT
r12_f04:--radius~1.2~--spectral-floor~0.04~$EOT
r25_f08_comp:--radius~2.5~--spectral-floor~0.08~--floor-gate~complement~$EOT
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
  echo "[lowdist] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
