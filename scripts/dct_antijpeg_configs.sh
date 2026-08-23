#!/usr/bin/env bash
# DCT-Shield **自己的**抗 JPEG 設定，這是先前漏測的一格。
#
# 它的抗壓縮保證是單向的（補充材料 D.4）：`Q_alg = q` 產生的影像**只在攻擊方
# 壓縮品質 q' >= q 時有效**。§5.4 的編輯任務用 0.95，而 §6.3 圖 6 的抗 JPEG
# 變體用的是 **Y-only ＋ Q_alg = 0.85**。
#
# 先前 `runs/ip2p_dct_band_extend` 的 Y-only 兩點用的是 `--q-alg 0.95`——當時的
# 理由是「讓兩支之間唯一的差別是通道集合」，那對畫失真曲線是對的，對抗淨化
# 是錯的：那個設定由構造就擋不住品質 75 的壓縮。於是本專案在 JPEG 上的優勢
# 是打在一個論文自己說擋不住 JPEG-75 的設定上。
#
# `docs/RESULTS.md` 早已記過同一件事（SDEdit 線上 dct_shield_y 在 jpeg75 拿
# +0.5185、本方法 +0.1349），並明寫「頭對頭表不可只放 base 變體」。
#
# 這一批把 q_alg 0.85 與 0.75 補齊，好在失真帶內取得可比的工作點；
# 抗淨化另外跑。
#
# 用法：bash scripts/dct_antijpeg_configs.sh "0 1 2 3 4 5 6"
# 七個點各佔一張卡，空卡少於七張時自動繞回（每卡多一個點就多約 2.1 小時）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_dct_antijpeg
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --dct-steps 1000"

DEVS=(${1:-0 1 2 3})

# 每卡最多兩個 process（`docs/OPERATIONS.md`：每個約 9 GB，3090 是 24 GB，
# 而卡是多人共用的）。**點數超過 卡數×2 時必須拒絕**，不可讓卡號公式繞回去
# 疊加——實測 `i/2 % 卡數` 在 6 點配 2 卡時把 4 個 process 塞到同一張卡上，
# 那一批六點掛掉四點，全部是 CUDA OOM。
require_slots() {
  local n_points="$1" n_devs="$2"
  if [ "$n_points" -gt $(( n_devs * 2 )) ]; then
    echo "錯誤：$n_points 個工作點需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
    echo "      只給了 $n_devs 張。每卡最多 2 個 process。" >&2
    exit 2
  fi
}


# tag:condition:q_alg:eps
POINTS="
y_q85_e10:dct_shield_y:0.85:1.0
y_q85_e14:dct_shield_y:0.85:1.4
y_q85_e20:dct_shield_y:0.85:2.0
y_q75_e10:dct_shield_y:0.75:1.0
y_q75_e14:dct_shield_y:0.75:1.4
base_q75_e10:dct_shield:0.75:1.0
base_q75_e14:dct_shield:0.75:1.4
"

require_slots "$(echo "$POINTS" | grep -c .)" "${#DEVS[@]}"
i=0
for p in $POINTS; do
  IFS=: read -r tag cond q eps <<< "$p"
  # **一格一張卡**（`i % N` 而不是 `i / 2 % N`）：這一批是決定論文重心的那
  # 一格，而 DCT-Shield 的 1000 步 PGD 是全專案最貴的單位工作（約 575 秒／
  # 張 × 13 張）。卡夠的話七個點各佔一張，關鍵路徑由約 4.2 小時降到約 2.1
  # 小時；卡不夠時 `i % N` 自動繞回，密度與原本相同。
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --conditions "$cond" --q-alg "$q" --eps "$eps" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[dct_antijpeg] $tag cond=$cond q_alg=$q eps=$eps dev=$dev pid=$!"
done
echo "[dct_antijpeg] 全部送出（$(date)）"
