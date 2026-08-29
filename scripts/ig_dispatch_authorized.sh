#!/usr/bin/env bash
# 四個影像引導工作點的直接派工。
#
# 為什麼不走 `image_guidance_converge.sh`
# ────────────────────────────────────────────────────────────────────
# 那支會呼叫 `free_cards.sh --assert`，而目前另一位使用者的**單一** process
# 在八張卡上各開了一個 256 MiB 的 CUDA context（實測 `--query-compute-apps`
# 只有一個 pid，每張 256 MiB，其中一張 4.1 GB）。守門依規則判定「卡上有別人
# 的 compute app」而全部拒絕。使用者已就本批次明確授權忽略那個 context。
#
# **這支不是常設的繞道**：它把授權寫死在檔頭，只給這一批用。守門本身沒有被
# 改動，其餘所有派工腳本照舊拒絕。要重複這種情況必須重新取得授權。
#
# 仍然保留的檢查：卡上**別人實際佔用的記憶體**若超過 1 GB 就拒絕——那代表
# 對方真的在算，不是只留了 context。
#
# 用法：bash scripts/ig_dispatch_authorized.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }

# 授權的範圍只到「忽略小的 context」，不到「擠掉真的在算的人」。
MINE=$(ps -u "$USER" -o pid= | tr -d ' ' | paste -sd'|' -)
for d in "${DEVS[@]}"; do
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
         | awk -F', ' -v i="$d" '$1==i {print $2}')
  other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
          --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u {print $2" "$3}' \
          | grep -vE "^($MINE) " | awk '{s+=$2} END {print s+0}')
  if [ "${other:-0}" -gt 1024 ]; then
    echo "錯誤：卡 $d 上別人實際用了 ${other} MiB（> 1 GB），拒絕啟動" >&2
    exit 3
  fi
  echo "卡 $d：別人佔用 ${other:-0} MiB（僅 context，授權範圍內）"
done

OUT=runs/ip2p_ig_converge
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)
BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

POINTS="ig_d21:diffuse_src:2.1 ig_d25:diffuse_src:2.5 ig_n30:noise:3.0 ig_n35:noise:3.5"

i=0
for p in $POINTS; do
  IFS=: read -r tag zt r <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE --loss image_guidance --ig-zt "$zt" --radius "$r" \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[ig] $tag dev=$dev zt=$zt r=$r"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 process"
