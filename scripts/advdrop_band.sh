#!/usr/bin/env bash
# AdvDrop 在 IP2P 線上的失真掃描，把失真帶夾在中間。
#
# 為什麼要跑它（`docs/RUN_QUEUE.md` 第四優先）：它是「統計驅動 vs 實現驅動」
# 那個假說（`SURVEY_ARCHITECTURE` 第三節 H1）的判定實驗，因為它**站在對角線
# 上**——與 DCT-Shield 共用 8×8 格點、與本方法共用非加性：
#
#                加性？          格點        裁切保留
#   DCT-Shield    加性           8×8          98.2%
#   AdvDrop       非加性（量化）  8×8          要量
#   本方法        乘法＋加性下限  32×32/hop 8  13%
#
# 若它抗裁切，決定強健度的是統計結構而非加性與否；若它不抗裁切，該假說垮。
# 兩種結果都有用。
#
# 這一支只跑防禦與失真，**抗淨化另外跑**：要先看失真帶落在哪兩點之間，
# 挑帶內的 1–2 點再送，否則會在出界的工作點上比。
#
# 用法：bash scripts/advdrop_band.sh "<兩個卡號>" ["<eps1 eps2 ...>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_advdrop_band
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

# **用 baseline 自己的損失與自己的步數**（CLAUDE.md：跑 baseline 時用它自己的
# 損失，換成本專案的損失是消融要分開列）。`--advdrop-steps` 與
# `--advdrop-step-size` 的預設就是重現時對到論文的那一組
# （步長 4，見 runs/advdrop_repro），這裡不覆寫。
COMMON="--data data/omniedit150 --conditions advdrop"

DEVS=(${1:-0 1})
# 四點把失真帶夾在中間（本方法工作點 DISTS 0.1043-0.1377）。可由參數換。
EPS_LIST=(${2:-60 100 150 220})
# 步數。**預設 50 是論文 §4.3 的未定向步數，而在這個威脅模型上它太少**：
# `runs/ip2p_advdrop_band/q_trajectory.csv` 量到 50 步跑完量化表只走到平均
# 39.3，而 eps = 220 的上界是 221——**貼上界的比例是 0.0000**，於是四個 eps
# 給出逐位相同的結果。500 是論文同一節給定向攻擊用的步數；由每步淨走 0.79
# 外推，屆時 q 平均約 390，eps 才會真的咬到。
#
# 換了步數就換輸出目錄（`runs/ip2p_advdrop_band_s500`），否則兩種步數的列會
# 混在同一個 tag 底下，而「誰比較強」與「誰跑比較久」在報表上就分不開了。
STEPS=${3:-50}
if [ "$STEPS" != "50" ]; then
  OUT="${OUT}_s${STEPS}"
  mkdir -p "$OUT"
fi

require_slots() {
  local n_points="$1" n_devs="$2"
  if [ "$n_points" -gt $(( n_devs * 2 )) ]; then
    echo "錯誤：$n_points 個工作點需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
    echo "      只給了 $n_devs 張。每卡最多 2 個 process。" >&2
    exit 2
  fi
}
# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋
# （實測踩過兩次：一次只讀 nvidia-smi 的前三行，一次等待器印了空卡清單卻沒有
# 依它決定要不要送）。任何一張指定的卡上有別人的 process 就直接拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

require_slots "${#EPS_LIST[@]}" "${#DEVS[@]}"

i=0
for eps in "${EPS_LIST[@]}"; do
  tag="e$(echo "$eps" | tr -d '.')"
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --advdrop-eps "$eps" --advdrop-steps "$STEPS" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[advdrop_band] $tag eps=$eps steps=$STEPS dev=$dev pid=$!"
done
echo "[advdrop_band] 全部送出（$(date)），共 $i 個 process"
