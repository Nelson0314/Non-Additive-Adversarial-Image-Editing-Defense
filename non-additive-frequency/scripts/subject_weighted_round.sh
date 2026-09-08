#!/usr/bin/env bash
# 把預算花在**主體的 latent token 上**：三個工作點，一格一卡。
#
# 這一輪的依據
# `runs/ig_probe/by_region_*.csv` 量到原圖的影像引導殘差有 **64% 落在受保護
# 主體自己的 latent token 上**，而主體只佔 53% 的面積；同一批也量到把主體
# 凍結的防禦一律卡在 0.41 的天花板。合起來的推論是：**能不能擋下取決於能不能
# 壓低主體那些 token 的殘差**，而均勻平均把將近一半的預算花在別處。
#
# 三格分別問三件事：
#
#   pg_subject   相位族 ＋ 主體加權。對照組是 `runs/ip2p_ig_loss/ig_noeot`
#                （同半徑、同步數、同收斂設定，只差權重）。
#   dct_ig       DCT-Shield 的**參數化** ＋ 本專案的 image_guidance 損失。
#                那個參數化目前的位移／失真比最好（DISTS 0.080 換到 0.676），
#                而那個損失在同失真下比 ‖E‖ 多壓四成殘差。兩者從未組合過。
#   dct_ig_subj  同上再加主體加權。
#
# 後兩格一律標 `modified_from_paper`：換掉損失就不是那篇的 baseline 了，
# `dct_loss` 欄逐列記下跑的是哪一個。
#
# 用法：bash scripts/subject_weighted_round.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_subject_weighted
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"

# 相位族的設定逐項取自 `scripts/ig_loss_round.sh` 的 `ig_noeot`，只差權重。
PHASE_BASE="--conditions phase_gain --quantile 0 --freq-weight jpeg_luma \
--freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 --steps 3000 --step-size 0.01"
# DCT-Shield 的設定逐項取自 `runs/ip2p_mainline/dct_native`（q_alg 0.95、
# eps 1.0、1000 步），那是位移／失真比最好的一格。
DCT_BASE="--conditions dct_shield --q-alg 0.95 --eps 1.0 --dct-steps 1000 \
--dct-loss project"

COMMON="--data data/omniedit150 --images $IMGS --loss image_guidance \
--ig-zt diffuse_src --eval-every 200 --eval-draws 8 --patience 10 \
--min-delta 0.0002 --save-weights"

POINTS="
pg_subject:$PHASE_BASE~--radius~2.5~--ig-weight~subject
dct_ig:$DCT_BASE
dct_ig_subj:$DCT_BASE~--ig-weight~subject
"

n=$(echo "$POINTS" | grep -c ':')
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

ONLY="${2:-}"
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  tag="${line%%:*}"; rest="${line#*:}"
  if [ -n "$ONLY" ]; then
    case "$tag" in $ONLY*) ;; *) continue ;; esac
  fi
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  # `~` 當分隔還原成空白：POINTS 每行是一個工作點，不能被 IFS 拆開
  args=$(echo "$rest" | tr '~' ' ')
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $COMMON $args \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[subjw] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"
