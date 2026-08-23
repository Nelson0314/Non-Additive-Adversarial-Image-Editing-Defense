#!/usr/bin/env bash
# DCT-Shield **自己的**抗 JPEG 設定，在抗淨化上的頭對頭。
#
# `scripts/dct_antijpeg_configs.sh` 只做前半段：它產生防禦圖與失真讀數，寫進
# `runs/ip2p_dct_antijpeg/<tag>`。抗淨化是另一支——`scripts/phase_retention.py`
# 讀**已存的**防禦圖，再跑一輪淨化加編輯。少了這一步，q_alg 0.85 的批次跑完
# 之後 JPEG 那一欄仍然是空的，而那一欄正是本方法唯一站得住的優勢軸。
#
# 為什麼非補不可：`runs/ip2p_purify_headtohead` 的 Y-only 點用的是
# `--q-alg 0.95`（§5.4 的編輯設定），而 DCT-Shield 的抗 JPEG 變體是 §6.3 的
# Y-only ＋ `Q_alg = 0.85`。它的抗壓縮保證是單向的（補充材料 D.4）：
# `Q_alg = q` 只在攻擊方壓縮品質 `q' >= q` 時有效，0.95 由構造就擋不住品質 75
# 的壓縮。在這一批跑完之前，本方法在 JPEG 上的優勢是打在一個論文自己說擋不住
# JPEG-75 的設定上，不可引用。
#
# **工作點由參數給，不寫死。** 頭對頭要的是等失真上可比的點，而 q0.85／q0.75
# 各點的 fDISTS 要等前半段跑完才知道；先看 `runs/ip2p_dct_antijpeg/*/results.csv`
# 挑落在本方法工作點失真帶裡的那一兩個，再送這一支。
#
# **空白地板不在這裡跑。** 地板與條件無關（防禦圖就是原圖），用
# `runs/ip2p_purify_headtohead/floor_*.csv` 那一份；`retention_table.py` 逐圖
# 去重再相減。這一支只檢查它在不在，不重算。
#
# 用法：
#     bash scripts/purify_antijpeg.sh "<卡號>" "y_q85_e14 y_q75_e14"
#
# 出表（`--src` 會遞迴走進 `antijpeg/` 與 `gridpure/`，地板在上一層）：
#     "$PY" scripts/retention_table.py \
#         --src runs/ip2p_purify_headtohead \
#         --out runs/ip2p_purify_headtohead/net_gain.csv
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
# 不 source ~/env.sh：它最後一行會把工作目錄換到舊的 ~/WACV（坑一）。
# 但 DIFFPURE_CKPT 只寫在那裡，少了它 gridpure 會被判為「相依不齊」而
# **靜默跳過**，報表上只剩一行提示。
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

HEAD=runs/ip2p_purify_headtohead
OUT="$HEAD/antijpeg"
SRCROOT=runs/ip2p_dct_antijpeg
mkdir -p "$OUT"

# 卡號由參數給，**不寫死**：卡是多人共用的。每張卡放兩個 process。
DEVS=(${1:-0 1 2})
# 工作點必填。沒有合理的預設——它取決於前半段量到的失真，猜一個等於偽造對齊。
TAGS=(${2:-})
if [ ${#TAGS[@]} -eq 0 ]; then
  echo "用法：$0 \"<卡號>\" \"<tag1> <tag2> ...\"" >&2
  echo "可選的 tag（先看它們的 fDISTS 再挑）：" >&2
  ls -1 "$SRCROOT" 2>/dev/null | sed 's/^/    /' >&2
  exit 2
fi

# 十三張人眼確認服從的影像（runs/ip2p_fair_comparison/images13.txt），
# 分片依任務族群切，不用流水號。
COLOR="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
SCENE="task_env_weather_112463 task_env_weather_246440 task_env_weather_63722"
OBJECT="task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

# 與 scripts/purify_headtohead.sh 同一組算子——換一組就不能跟現有的列並排。
# impress 佔一格約 82% 的機時，兩支都另外補跑。
PUR="identity blur1 jpeg75 jpeg30 crop_resize0.1 jpeg_then_resize75 adverse_cleaner gridpure"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

# 前半段沒跑完的 tag 一律報錯，不跳過：那是打錯字或批次沒跑完，兩者都不該
# 讓整批安靜地少一格。
#
# **查的是防禦圖不是 results.csv**：`ip2p_run.py` 逐張寫出 CSV，跑到第三張時
# 檔案就已經存在，只查它會讓這一支在防禦圖還沒齊的時候送出去，而
# `phase_retention.py` 要到跑進那一張才 `FileNotFoundError`——一片跑了一個
# 多小時才死。十三張都在才算數。
for tag in "${TAGS[@]}"; do
  if [ ! -d "$SRCROOT/$tag" ]; then
    echo "[antijpeg] 沒有 $SRCROOT/$tag——tag 打錯或前半段沒送出" >&2
    exit 3
  fi
  n=$(ls -1 "$SRCROOT/$tag"/*__def.png 2>/dev/null | wc -l)
  if [ "$n" -ne 13 ]; then
    echo "[antijpeg] $tag 只有 $n/13 張防禦圖——前半段（dct_antijpeg_configs.sh）還沒跑完" >&2
    exit 3
  fi
done

# 地板缺席的話淨增益整張表算不出來（逐圖相減，不補零）。先報覆蓋率再送。
shopt -s nullglob
FLOORS=("$HEAD"/floor_*.csv)
shopt -u nullglob
if [ ${#FLOORS[@]} -eq 0 ]; then
  echo "[antijpeg] $HEAD 底下沒有 floor_*.csv——空白地板不可省略（DEC），先跑地板" >&2
  exit 4
fi
NFLOOR=$(tail -q -n +2 "${FLOORS[@]}" | cut -d, -f1 | sort -u | wc -l)
echo "[antijpeg] 地板現有 $NFLOOR/13 張（${#FLOORS[@]} 個分片）；不足的格子 retention_table.py 會排除並回報"

i=0
launch() {   # $1 tag  $2 shard  $3 imgs
  local dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
      --run "$SRCROOT/$1" --images $3 $COMMON --out "$OUT/$1_$2.csv" \
      > "$OUT/$1_$2.log" 2>&1 &
  echo "[antijpeg] $1/$2 dev=$dev pid=$!"
}

for tag in "${TAGS[@]}"; do
  launch "$tag" color  "$COLOR"
  launch "$tag" scene  "$SCENE"
  launch "$tag" object "$OBJECT"
done
echo "[antijpeg] 全部送出（$(date)），共 $i 個 process"
