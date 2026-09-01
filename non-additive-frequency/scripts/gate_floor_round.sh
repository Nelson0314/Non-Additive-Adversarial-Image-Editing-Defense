#!/usr/bin/env bash
# 紋理閘 × 頻譜加性下限的 2×4 全因子。
#
# 為什麼
# ────────────────────────────────────────────────────────────────────
# 把瑪利歐那張切成 16×16 小塊、按亮度變異數分成「平坦一半」與「有內容一半」，
# 量殘差能量落在平坦那一半的比例：`r25` 5.5%、`r80` 4.3%、`r40_eot` 8.7%、
# `floor04_eot` 18.5%。**一半的畫面只吃到 4–9% 的擾動**，殘差圖把主體整個
# 描出來，牆是黑的。
#
# 主因是**乘性參數化**不是閘：`|ΔS| = 2A·sin(θ/2)`，`A` 是原圖自己的振幅，
# 平坦區 `A ≈ 0`，閘全開也動不了（`docs/METHOD.md` 構造限制第一條）。
#
# 但閘做的事正好加重症狀。`--quantile 0` 讓能量因子恆為 1，閘只剩
# `(1 − coh²)^p`；它壓的是 coherence 高的地方，也就是**清晰的輪廓**，放行的
# 是梯度方向雜亂的地方，也就是**臉的明暗、布料織紋**。於是擾動被從輪廓推進
# **主體內部的細節**，而那正是人眼最容易看出來的位置。
#
# 兩個軸因此各對準一半：
#   `--gate-edge-power 0`  把閘的唯一活項關掉。**這是方法的簡化，少一個元件。**
#                          `docs/METHOD.md` 標「待測」：0 與 0.5 在較高失真上
#                          擋下率較高，等失真對照從未做過。
#   `--spectral-floor`     唯一能把擾動帶到平坦區的旋鈕。0.04 已測到 18.5%，
#                          0.08 與 0.12 從未跑過。
#
# 全部帶 `eot_broad`（模糊族含 3.0，讓評測用的 σ=2 由邊界變成內點），
# 損失 `latent_norm`（只過 VAE 編碼器，不碰 UNet），半徑固定 4.0。
# 影像兩張（使用者指定）：盆栽人與瑪利歐。
#
# 用法：bash scripts/gate_floor_round.sh "<四個以上卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 4 ] && { echo "用法：$0 \"<四個以上卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_gate_floor
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"
EOT="--purify-aware~eot_broad~--eot-sigmas~0.5~1.0~2.0~3.0"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss latent_norm --radius 4.0 --steps 1000 --step-size 0.01 \
--save-weights --skip-existing"

# g1 = 閘開（現行 `--gate-edge-power 1`）、g0 = 閘關。f 後面是加性下限。
POINTS="
g1_f00:--gate-edge-power~1.0~$EOT
g0_f00:--gate-edge-power~0~$EOT
g1_f04:--gate-edge-power~1.0~--spectral-floor~0.04~$EOT
g0_f04:--gate-edge-power~0~--spectral-floor~0.04~$EOT
g1_f08:--gate-edge-power~1.0~--spectral-floor~0.08~$EOT
g0_f08:--gate-edge-power~0~--spectral-floor~0.08~$EOT
g1_f12:--gate-edge-power~1.0~--spectral-floor~0.12~$EOT
g0_f12:--gate-edge-power~0~--spectral-floor~0.12~$EOT
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
  echo "[gatefloor] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
