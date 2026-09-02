#!/usr/bin/env bash
# 只在**主體之外**可學的色彩網格：兩個半徑 × 兩張影像，一格一卡，跑滿。
#
# 遮罩由 CLIPSeg 依 `data/decoy_catalogue.yaml` 的 `objects` 產生
# （`src/defense/subject_mask.py`），`apply_where = 1 − 遮罩`，
# 於是主體區域**逐位元**不動，擾動只長在背景。核心面積：盆栽人 44.5%、
# 瑪利歐 49.1%，故可動的面積約各剩五成。
#
# 損失取 `image_guidance`：`runs/latent_norm_probe/` 證實 `latent_norm` 在
# 色彩族裡與位移反向（等失真對隨機 0.86），換成 `image_guidance` 之後
# 未淨化欄變 1.08–1.14。`--ig-zt diffuse_src` 沿用專案既有的全部 IG 批次。
#
# 半徑取 0.10 與 0.20：可動面積少了一半，同半徑的失真也隨之下降
# （實測 r=0.10 帶遮罩是 DISTS 0.1023，不帶是 0.1660），故要往上挪才落在
# 可比較的帶上。兩點畫得出取捨曲線，等失真內插才做得成。
#
# 八千步、每兩百步以固定抽樣評估、連續八次無改善才停。IG 的三個無遮罩工作點
# 都跑滿八千步未觸發早停，這幾格照同一個預算跑，停止原因逐列記在 CSV。
#
# 用法：bash scripts/color_masked_sweep.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_color_masked
mkdir -p "$OUT"

COMMON="--data data/omniedit150 --conditions color_grid --loss image_guidance \
--ig-zt diffuse_src --steps 8000 --eval-every 200 --patience 8 --save-weights \
--subject-mask data/decoy_catalogue.yaml"

CELLS="m_r010_plant:0.10:task_attr_mod_color_11699 \
m_r010_mario:0.10:task_attr_mod_color_6205 \
m_r020_plant:0.20:task_attr_mod_color_11699 \
m_r020_mario:0.20:task_attr_mod_color_6205"

ONLY="${2:-}"
n=$(echo $CELLS | wc -w)
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 格需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
for cell in $CELLS; do
  tag="${cell%%:*}"; rest="${cell#*:}"
  rad="${rest%%:*}"; img="${rest##*:}"
  if [ -n "$ONLY" ]; then
    case "$tag" in $ONLY*) ;; *) continue ;; esac
  fi
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --radius "$rad" --images "$img" $COMMON \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[masked] $tag radius=$rad image=$img dev=$dev"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"
