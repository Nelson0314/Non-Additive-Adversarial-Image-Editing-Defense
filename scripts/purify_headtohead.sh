#!/usr/bin/env bash
# 抗淨化的頭對頭：兩個主線設定 對 DCT-Shield 的三個工作點。
#
# **只讀已存的防禦圖，不重跑防禦**（scripts/phase_retention.py）。
# 空白地板不可省略：淨化算子自己就會把編輯推開，不扣掉它，「淨化後位移較大」
# 無法排除「該算子本來就把編輯推得比較開」這個平庸解釋。
#
# 五個條件的失真並不相等（DCT-Shield 在 DISTS 上便宜得多），故頭對頭表要與
# matched_distortion_table.py 的等失真內插一起讀，不可單看絕對位移。
#
# 分片依任務族群切，不用流水號：color（5 張）、scene（3 張）、object（5 張）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
# 不 source ~/env.sh：它最後一行會把工作目錄換到舊的 ~/WACV（坑一）。
# 但 DIFFPURE_CKPT 只寫在那裡，少了它 gridpure／fdpure 會被判為
# 「相依不齊」而**靜默跳過**，報表上只剩一行提示。
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_purify_headtohead
mkdir -p "$OUT"

COLOR="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
SCENE="task_env_weather_112463 task_env_weather_246440 task_env_weather_63722"
OBJECT="task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"
ALL="$COLOR $SCENE $OBJECT"

# 八個算子。impress 佔一格約 82% 的機時，另外補跑。
PUR="identity blur1 jpeg75 jpeg30 crop_resize0.1 jpeg_then_resize75 adverse_cleaner gridpure"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

# tag:防禦圖所在目錄
SRC="
ours_nonadd:runs/ip2p_overlap_sweep/h08_r25
ours_add:runs/ip2p_axis_necessity/b_pg_r20
dct_e14:runs/ip2p_band_calibration/dct_e14
dct_e18:runs/ip2p_dct_band_extend/dct_e18
dct_y_e14:runs/ip2p_dct_band_extend/dct_y_e14
"

# 卡號由參數給，**不寫死**：卡是多人共用的。十八個 process（五個條件 × 三片
# ＋ 三片地板）每張卡放兩個，八張卡剛好放滿；先前寫死成 `i / 2`，索引會走到
# 8，而機器只有 0–7。用法 `bash scripts/purify_headtohead.sh "0 1 2 3 4 5 6 7"`。
DEVS=(${1:-0 1 2 3 4 5 6 7})
# 要送哪一半。條件格與地板的完成度常常不同步——條件那 15 格已經 13/13，地板
# 只到 6/13，整批重送等於白跑 15 個 process。
#     all         條件格 ＋ 地板（預設）
#     conditions  只送五個條件 × 三片
#     floor       只送三片空白地板
PARTS="${2:-all}"
case "$PARTS" in all|conditions|floor) ;; *)
  echo "用法：$0 \"<卡號>\" [all|conditions|floor] [\"<分片>\"]" >&2; exit 2 ;;
esac
# 要跑哪幾片。分片名只能是 color / scene / object——`retention_table.py` 的
# `tag_of()` 由檔名還原條件標籤，別的名字會拋錯。給子集是為了補跑：地板的
# 三片完成度常常不同，重跑已經跑完的那一片不但白花機時，`write_csv` 還會把
# 那一片既有的列整個蓋掉，中途再斷一次就比現在更少。
SHARDS=(${3:-color scene object})
for sh in "${SHARDS[@]}"; do
  case "$sh" in color|scene|object) ;; *)
    echo "錯誤：分片名只能是 color / scene / object，收到 $sh" >&2; exit 2 ;;
  esac
done

# 每卡最多兩個 process（`docs/OPERATIONS.md`）。**launch 數超過 卡數×2 時
# 必須拒絕**，不可讓卡號公式繞回去疊加——實測疊到四個就整批 CUDA OOM。
require_slots() {
  local n_points="$1" n_devs="$2"
  if [ "$n_points" -gt $(( n_devs * 2 )) ]; then
    echo "錯誤：$n_points 個 process 需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
    echo "      只給了 $n_devs 張。每卡最多 2 個 process。" >&2
    exit 2
  fi
}

# 條件格 = 有防禦圖的來源數 × 分片數；地板 = 分片數。
n_src=0
for s in $SRC; do
  IFS=: read -r _ run <<< "$s"
  [ -f "$run/results.csv" ] && n_src=$(( n_src + 1 ))
done
n_cond=0; n_floor=0
[ "$PARTS" != floor ] && n_cond=$(( n_src * ${#SHARDS[@]} ))
[ "$PARTS" != conditions ] && n_floor=${#SHARDS[@]}
# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋
# （實測踩過兩次：一次只讀 nvidia-smi 的前三行，一次等待器印了空卡清單卻沒有
# 依它決定要不要送）。任何一張指定的卡上有別人的 process 就直接拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

require_slots "$(( n_cond + n_floor ))" "${#DEVS[@]}"

i=0
launch() {   # $1 tag  $2 run  $3 shard  $4 imgs  $5 extra
  local dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
      --run "$2" --images $4 $COMMON $5 --out "$OUT/$1_$3.csv" \
      > "$OUT/$1_$3.log" 2>&1 &
  echo "[purify] $1/$3 run=$2 dev=$dev pid=$!"
}

for s in $SRC; do
  [ "$PARTS" = floor ] && break
  IFS=: read -r tag run <<< "$s"
  if [ ! -f "$run/results.csv" ]; then
    echo "[purify] 跳過 $tag：$run/results.csv 不存在"; continue
  fi
  for shard in "${SHARDS[@]}"; do
    case $shard in
      color) imgs="$COLOR" ;; scene) imgs="$SCENE" ;; object) imgs="$OBJECT" ;;
    esac
    launch "$tag" "$run" "$shard" "$imgs" ""
  done
done

# 空白地板：防禦圖就是原圖，與條件無關，故只跑一份——但**仍要分片**。
# 不分片的話它一個 process 要跑十三張，而其餘每個分片只跑三到五張；扣地板
# 的表要等最慢的那一格，於是整批的完成時間由這一格決定（實測差四小時）。
for shard in "${SHARDS[@]}"; do
  [ "$PARTS" = conditions ] && break
  case $shard in
    color) imgs="$COLOR" ;; scene) imgs="$SCENE" ;; object) imgs="$OBJECT" ;;
  esac
  launch floor runs/ip2p_overlap_sweep/h08_r25 "$shard" "$imgs" "--floor"
done

echo "[purify] 全部送出（$(date)），共 $i 個 process"
