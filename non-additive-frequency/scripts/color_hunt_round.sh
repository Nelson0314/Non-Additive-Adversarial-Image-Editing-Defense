#!/usr/bin/env bash
# 色彩族的方向探勘：三個從未跑過的組合，一格一卡，快速看有沒有效。
#
# 這一輪的依據
# `runs/ig_probe/by_region_*.csv` 量到影像引導殘差的 **64% 落在受保護主體的
# latent token 上**，而主體只佔 53% 的面積——均勻平均把將近一半的預算花在
# 對「主體會不會被編輯」影響較小的地方。色彩族最缺的正是效率（同樣的效果
# 要付兩倍失真），所以把預算重新分配是這一族最直接可用的一件事。
#
# 三格：
#
#   cg_subj_r010  色彩網格 ＋ 主體加權。對照 `runs/ip2p_color_converge/
#                 image_guidance_*`（同半徑、同損失，只差權重）。
#   cg_subj_r020  同上，較大半徑。對照 `runs/ip2p_color_ig/ig_r020_*`。
#   cc_ig_r3      **色彩曲線族 ＋ image_guidance**。曲線族至今只跑過
#                 `latent_norm`，而那個損失已被證實在這一族裡幾乎沒用
#                 （同失真對隨機只多壓 7–9%，IG 是四成）。曲線族**裁切等變
#                 是精確的**（全域逐通道曲線，不含空間項），若它在 IG 底下
#                 站得住，抗裁切那一欄會比網格族好看。
#
# **一律不加遮罩。** 遮罩版已量到 0.41 的天花板（`runs/patch_probe/README.md`
# 第三節），這一輪要的是效率不是「主體逐位元不動」。
#
# 步數取 4000 ＋ 早停：這一輪是篩方向，不是定稿。有效的格再用
# `--resume-weights` 續跑到收斂。
#
# 用法：bash scripts/color_hunt_round.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_color_hunt
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"

COMMON="--data data/omniedit150 --images $IMGS --loss image_guidance \
--ig-zt diffuse_src --steps 4000 --eval-every 200 --eval-draws 8 \
--patience 8 --min-delta 0.002 --save-weights"

POINTS="
cg_subj_r010:--conditions~color_grid~--radius~0.10~--ig-weight~subject
cg_subj_r020:--conditions~color_grid~--radius~0.20~--ig-weight~subject
cc_ig_r3:--conditions~color_curve~--radius~3.0
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
  args=$(echo "$rest" | tr '~' ' ')
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $COMMON $args \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[colour] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"
