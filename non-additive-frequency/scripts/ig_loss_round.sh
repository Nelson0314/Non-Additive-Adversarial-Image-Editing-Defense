#!/usr/bin/env bash
# 換損失：從打 VAE 編碼器改成打 UNet 的影像引導項。
#
# 為什麼
# ────────────────────────────────────────────────────────────────────
# 今天所有工作點的損失都是 `latent_norm`（把 `‖E(x)‖₂` 壓小）。編碼器是淺層
# 卷積網路，反應由**局部紋理**主導——而局部紋理正是淨化器移除的東西。量到的
# 處境是：未淨化的淨增益 0.577（佔可達範圍 0.772 的 75%），模糊 σ1 只剩 23%、
# σ2 剩 9%、裁切剩 13%。動頻帶只把 σ2 由 0.052 挪到 0.067（1.3 倍），
# 動強度在 radius 8 就飽和（radius 16 的 DISTS／RMS 到小數第四位相同）。
#
# 紀錄裡最大的那個差距沒有被動過：`image_guidance`（讀 UNet）在模糊 σ1 拿到
# **0.322**，是所有 `latent_norm` 結果的兩倍以上。先前以「太慢」排除它，
# 那個理由在**兩張影像**上不再成立。
#
# 損失本身
# ────────────────────────────────────────────────────────────────────
#     L = ‖ε(z_t, E_img(x'), ∅) − ε(z_t, 0, ∅)‖²   ，**要最小化**
#
# 也就是要求 IP2P 取樣式裡 `s_I·[ε(z_t, c_I, ∅) − ε(z_t, 0, ∅)]` 這一項消失
# ——攻擊方的影像條件失去作用。`latent_norm` 是它的逐點版本（把影像條件推向
# 零張量），本項只要求 UNet 的**反應**相同，可行集大得多。
#
# 八個點分離四件事
# ────────────────────────────────────────────────────────────────────
#   `ln_eot`    對照：同設定、同 EOT，只是損失仍是 latent_norm
#   `ig_noeot`  新損失但不放 EOT ── 分離「損失」與「EOT」各自的貢獻
#   `ig_eot` / `ig_r40_eot`  新損失 ＋ EOT，兩個強度
#   `ig_blend_eot`  兩個損失相加（`--latent-norm-weight 1.0`，舊項已正規化故
#                   權重 1 真的是等權）。**程式早就寫好，從來沒跑過**；
#                   實測兩者是分工的——舊的買未淨化強度，新的買抗淨化
#   `ig_hit_eot` / `ig_lot_eot`  時間步窗的高噪聲端與低噪聲端。
#                   DiffusionGuard（ICLR 2025）整個目標就在 `t = T`，而它是
#                   唯一報得出裁切強健性的方法；查證後那個強健性來自損失，
#                   方法裡沒有任何空間不變性設計。這兩格把該機制單獨拿出來測
#   `ig_f08_eot`  ＋ 頻譜加性下限 0.08。今天量到那是唯一能把擾動從主體挪到
#                 平坦區的旋鈕（平坦半邊吃到的能量 8.7% → 29.0%）
#
# 收斂
# ────────────────────────────────────────────────────────────────────
# **隨機目標的逐步損失是取樣變異不是收斂訊號**（本專案犯過一次）。一律開
# `--eval-every`，用一組固定的 (t, eps) 評估並寫進 `trace.csv`，收斂看那一欄。
#
# 影像兩張（使用者指定）：盆栽人與瑪利歐。
#
# 用法：bash scripts/ig_loss_round.sh "<四個以上卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 4 ] && { echo "用法：$0 \"<四個以上卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_ig_loss
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"
EOT="--purify-aware~eot_broad~--eot-sigmas~0.5~1.0~2.0~3.0"
IG="--loss~image_guidance~--ig-zt~diffuse_src"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--steps 3000 --step-size 0.01 \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002 \
--save-weights --skip-existing"

POINTS="
ln_eot:--loss~latent_norm~--radius~4.0~$EOT
ig_noeot:$IG~--radius~2.5
ig_eot:$IG~--radius~2.5~$EOT
ig_r40_eot:$IG~--radius~4.0~$EOT
ig_blend_eot:$IG~--radius~2.5~--latent-norm-weight~1.0~$EOT
ig_hit_eot:$IG~--radius~2.5~--ig-t-min~800~--ig-t-max~1000~$EOT
ig_lot_eot:$IG~--radius~2.5~--ig-t-min~1~--ig-t-max~300~$EOT
ig_f08_eot:$IG~--radius~2.5~--spectral-floor~0.08~$EOT
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
  echo "[igloss] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
