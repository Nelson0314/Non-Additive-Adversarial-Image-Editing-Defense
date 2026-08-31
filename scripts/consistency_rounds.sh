#!/usr/bin/env bash
# 一致性那一條的兩輪：先減少冗餘（`--hop`），再軟懲罰（`--consistency-weight`）。
#
# 診斷結果（`runs/stft_consistency/`）
# ────────────────────────────────────────────────────────────────────
# 由已存的權重重建，量「要求的頻譜」與「重疊相加交得出來的」差多少：
#
#   ig_d25   amp_dev 0.175  phase_rho 0.882  投影前像素 L∞ 1.09
#   po_r45   amp_dev 0.304  phase_rho 0.854  投影前像素 L∞ 3.21
#   po_r60   amp_dev 0.480  phase_rho 0.834  投影前像素 L∞ 6.03
#
# 要求的頻譜有 18–48% 的幅度交不出來、相位丟掉 11–17%，而投影前的像素值域
# 遠超出合法的 0–1——要求的那個東西不是一張影像。四個 `po_*` 的 amp_dev
# 排序與人眼看到的醜的排序一位不差。
#
# 兩輪各打一半
# ────────────────────────────────────────────────────────────────────
# 甲（`ROUND=hop`）**從構造上減少矛盾**。block 32、hop 8 是 16 倍冗餘，每個
# 像素由 16 個視窗加總；hop 16 是 4 倍，hop 32 是臨界取樣——**此時任意相位
# 都可實現，amp_dev 構造上為 0**。代價是視窗之間不再平滑，區塊邊界可能自己
# 變成一種假影，那正是要看的。`hp_08` 是基準線（現行值）。
#
# 乙（`ROUND=cons`）**軟懲罰**。損失加上 w × 相對幅度偏差，於是同樣損失值的
# 解裡偏好可實現的那些。**與 `--gl-iters` 不同**：那是前向硬投影，會把效果
# 一起投影掉（FND-051）；這是偏好，換多少由 w 決定。`cw_000` 是基準線。
#
# 兩輪的其餘旗標與 `ig_d25` 完全相同，故三者可直接對照。
# **w 與 hop 的值都是本專案指定的，沒有出處。**
#
# 用法：ROUND=hop bash scripts/consistency_rounds.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：ROUND=hop|cons $0 \"<四個卡號>\"" >&2; exit 2; }
ROUND="${ROUND:-}"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"

case "$ROUND" in
  hop)
    OUT=runs/ip2p_hop_overlap
    POINTS="
hp_08:--hop~8
hp_12:--hop~12
hp_16:--hop~16
hp_32:--hop~32
"
    ;;
  cons)
    OUT=runs/ip2p_consistency
    POINTS="
cw_000:--consistency-weight~0
cw_003:--consistency-weight~0.03
cw_010:--consistency-weight~0.10
cw_030:--consistency-weight~0.30
"
    ;;
  *) echo "錯誤：ROUND 必須是 hop 或 cons" >&2; exit 2 ;;
esac

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
for flag in --consistency-weight; do
  grep -q -- "$flag" scripts/ip2p_run.py || {
    echo "錯誤：scripts/ip2p_run.py 不認得 $flag，先同步本機的改動" >&2; exit 2; }
done
mkdir -p "$OUT"

# `--hop` 在 BASE 裡也出現，後面再給一次會覆蓋（argparse 取最後一個）。
BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[$ROUND] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
