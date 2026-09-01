#!/usr/bin/env bash
# 色彩重映射的強度掃描：五個最佳化的工作點，配七個同族的隨機對照。
#
# 兩族的構造見 `src/defense/color_param.py`：
#
#   color_curve   逐通道 K 段單調分段線性曲線，全域。3K = 192 個參數。
#                 對裁切**精確等變**，輸出值域由構造落在 [0,1]、不需 clamp。
#   color_grid    雙邊網格上的仿射色彩變換（空間 8×8 × 亮度 8 格）。
#                 12·D·G² = 6144 個參數。有偏移項，故暗部也推得動。
#
# 半徑的選點取自 `runs/color_field_cost/`（本機、不需 GPU 的失真校準）。
# 那一批量的是**同半徑隨機場**的失真，是上界；最佳化的會落在它下面
# （`runs/ip2p_shading` 實測同半徑下最佳化 0.0165 對隨機 0.0354）。
# 隨機場的中位數 DISTS：
#
#   color_curve  r=0.25→0.0044  r=1→0.0210  r=2→0.0317  r=3→0.0363
#   color_grid   r=0.02→0.0111  r=0.05→0.0427  r=0.1→0.0949  r=0.2→0.1659
#
# **曲線族在失真軸上是有天花板的**：r 由 2 加到 3 只從 0.0317 走到 0.0363。
# 全域單調映射就是付不出結構性失真。故它與相位法工作點（DISTS 0.083）的
# 等失真頭對頭會落在範圍外，協定規定回報 `out_of_range` 而不是外插。
#
# 收斂：`--eval-every 100 --patience 5`，用一組**固定**的 (t, eps) 抽樣評估並
# 寫進 trace.csv。逐步損失是取樣變異不是收斂訊號，判收斂一律看那一欄。
#
# `--save-weights`：色彩族的參數要留著，才能在讀出端重新套用同一個變換
# （`effect_codefense = LPIPS(編輯(D(x)), D(編輯(x)))` 的右側）。防禦圖
# 不可逆推回參數。
#
# 用法：bash scripts/color_sweep.sh "<卡號…>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\"" >&2; exit 2; }
# 卡是多人共用的。**這個檢查會 exit，不是印出來就算。**
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_color
mkdir -p "$OUT"

# 六張，六類任務各一張。前五張逐字取自 `runs/ip2p_shading` 的十三張，
# 於是色彩族與明暗場可以逐圖相減。
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205 task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_add_13726"

COMMON="--data data/omniedit150 --loss latent_norm --steps 1000 --eval-every 100 --patience 5 --save-weights"

# tag:condition:radius
# 五個最佳化的工作點，剛好一卡一個。網格族給三個半徑（它的失真範圍橫跨
# 整個可用帶，等失真內插要三點才畫得出曲線）；曲線族給兩個端點（它的失真
# 天花板只到 DISTS 0.036，中間再插一點沒有新資訊）。
OPT_POINTS="curve_r1:color_curve:1.0 curve_r3:color_curve:3.0 grid_r005:color_grid:0.05 grid_r010:color_grid:0.10 grid_r020:color_grid:0.20"
# 隨機對照的半徑刻意跨得比最佳化的寬：等失真對齊是在對方的曲線上**內插**，
# 落在掃描範圍外一律回報 `out_of_range`，不外插。
RAND_POINTS="curve_rand_r025:color_curve_rand:0.25 curve_rand_r1:color_curve_rand:1.0 curve_rand_r3:color_curve_rand:3.0 grid_rand_r002:color_grid_rand:0.02 grid_rand_r005:color_grid_rand:0.05 grid_rand_r010:color_grid_rand:0.10 grid_rand_r020:color_grid_rand:0.20"

launch() {                      # $1 tag:cond:radius  $2 卡號
  local point="$1" dev="$2"
  local tag="${point%%:*}" rest="${point#*:}"
  local cond="${rest%%:*}" rad="${rest##*:}"
  CUDA_VISIBLE_DEVICES="$dev" "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --conditions "$cond" --radius "$rad" \
      --images $IMGS $COMMON < /dev/null > "$OUT/$tag.log" 2>&1
}

# 每卡最多 2 個 process（`docs/OPERATIONS.md`）。疊到 3 個實測整批 OOM。
n_opt=$(echo $OPT_POINTS | wc -w)
if [ "$n_opt" -gt $(( ${#DEVS[@]} * 2 - 1 )) ]; then
  echo "錯誤：$n_opt 個最佳化工作點加上一條隨機串超過每卡 2 個的上限，" >&2
  echo "      ${#DEVS[@]} 張卡最多容得下 $(( ${#DEVS[@]} * 2 - 1 )) 個。" >&2
  exit 2
fi

i=0
for point in $OPT_POINTS; do
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  setsid nohup bash -c "$(declare -f launch); PY='$PY'; OUT='$OUT'; IMGS='$IMGS'; COMMON='$COMMON'; launch '$point' '$dev'" \
      < /dev/null > /dev/null 2>&1 &
  disown
  echo "[color] $point dev=$dev（最佳化）"
done

# 隨機對照的 `params()` 為空，`run_param_pgd` 在 `if not ps` 那一行直接跳過
# 迴圈，所以一格只花一次編輯的時間。**全部串成一條**跑在同一張卡上，
# 免得某張卡同時扛三個 process。放在第二張卡，那張卡於是有兩個
# process，其餘各一個。
RAND_DEV=${DEVS[$(( 1 % ${#DEVS[@]} ))]}
setsid nohup bash -c "$(declare -f launch); PY='$PY'; OUT='$OUT'; IMGS='$IMGS'; COMMON='$COMMON'; for p in $RAND_POINTS; do launch \"\$p\" '$RAND_DEV'; done" \
    < /dev/null > "$OUT/rand_chain.log" 2>&1 &
disown
echo "[color] 隨機對照串成一條在 dev=$RAND_DEV"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process（$(date)）"
