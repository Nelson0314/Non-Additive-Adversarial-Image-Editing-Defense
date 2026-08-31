#!/usr/bin/env bash
# 探針：把 latent 推**離**分布，載體用極低頻的全域明暗場。
#
# 一張影像、兩張卡、四個點。這是設計探針不是主批次。
#
# 兩個想法，各有一半已經在專案裡
# ────────────────────────────────────────────────────────────────────
# **「拉到怪異目標」已經測過，而且方向是反的。** `--loss encoder_target` 就是
# 把 latent 拉向一個目標（灰圖）。同 13 張、同設定、只換損失：`encoder_target`
# 擋下 5/13、`latent_norm` 擋下 10–13/13，而單位失真換到的位移兩邊一樣
# （3.4 對 3.4）。機制寫在 `docs/RESULTS.md`：灰圖 latent 是一個「平坦灰內容」
# 的**有效** latent，UNet 照著它畫，輸出是**劣化**；零 latent 讓影像條件沒有
# 資訊，IP2P 退化成純文生圖，把 prompt 畫出來，那才是**重畫**。
#
# **但零不是怪異的點，是最熟悉的那一點。** IP2P 用 classifier-free guidance
# 訓練，訓練時本來就會隨機丟掉影像條件，所以 `c_I = 0` 是模型受訓過的狀態。
# 真正沒被訓練過的是**模長遠大於正常值**的條件。`--loss latent_norm_max`
# 就是把符號反過來，從未跑過。
#
# **「推離流形」對本輪要擋的三個算子其實不咬。** `--manifold-weight` 已經存在
# 但做的是相反的事（把交付圖拉向淨化器的不動點）。而「離開流形會被推回來」
# 只對 DiffPure／IMPRESS／GrIDPure 成立——它們才是流形投影；JPEG、模糊、裁切
# 是固定的線性算子與量化，不把影像拉回自然統計。
#
# 真正的阻力在頻帶：離流形要放異常能量，放高頻被模糊砍掉，放低頻模糊砍不掉但
# 編碼器在低頻的位移／DISTS 是 330、中高頻是 24772（`RESULTS.md` 頻帶表）。
#
# 低頻場為什麼仍值得一試
# ────────────────────────────────────────────────────────────────────
# `SURVEY_ARCHITECTURE` §0.2 證明「裁切贏不了」，前提是**效果依賴一個空間對齊
# 的圖樣**（相位誤差 `2π·f·0.2488·r` 要小於 π/2）。**全域的低階統計沒有對齊
# 可破**——裁掉外圈再放大，整體色偏／照明異常原封不動。那是該證明沒有涵蓋的
# 唯一情況。
#
# 而它不貴：`runs/shading_field_cost` 實測 RMS 0.06 的乘性明暗場付 DISTS
# 0.0884，每單位 RMS 1.47，低於本方法工作點的 2.30。**那一格從沒被否決過效果
# ——它只量了失真，「換得到多少位移」需要 GPU，從來沒跑。**
#
# 四個點
# ────────────────────────────────────────────────────────────────────
#   `ph_lnmax`     現行參數化 ＋ 最大化：把 latent 推出分布會怎樣
#   `shade_lnmax`  低頻明暗場 ＋ 最大化：候選二的步驟 1
#   `shade_lnmin`  低頻明暗場 ＋ 壓到零：損失方向的對照
#   `shade_rand`   同半徑的隨機明暗場，**不最佳化**。候選二第 6 點步驟 3 標為
#                  必做——`位移場`（FND-004）的死法正是「與同失真隨機對照
#                  無法區分」，低自由度的參數化特別容易重蹈覆轍
#
# 用法：bash scripts/offmanifold_probe.sh "<兩個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 2 ] && { echo "用法：$0 \"<兩個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
grep -q "latent_norm_max" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 latent_norm_max，先 git pull" >&2; exit 2; }

OUT=runs/ip2p_offmanifold
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699"

BASE="--data data/omniedit150 --quantile 0 --freq-weight jpeg_luma \
--freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--steps 1000 --step-size 0.01 --save-weights --skip-existing"

# 明暗場的半徑是粗網格上 log 增益的 L∞ 界，上界 0.30（再大亮部整片飽和，
# clamp 之後多出來的預算不會變成擾動）。取 0.25 留一點餘裕。
POINTS="
ph_lnmax:--conditions~phase_gain~--loss~latent_norm_max~--radius~4.0
shade_lnmax:--conditions~shading~--loss~latent_norm_max~--radius~0.25
shade_lnmin:--conditions~shading~--loss~latent_norm~--radius~0.25
shade_rand:--conditions~shading_rand~--loss~latent_norm~--radius~0.25
"

n_points=$(echo "$POINTS" | grep -c ':')
# 每卡最多 2 個 process（`docs/OPERATIONS.md`）。疊到 3 個實測整批 OOM。
if [ "$n_points" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n_points 個 process 需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
  echo "      只給了 ${#DEVS[@]} 張。每卡最多 2 個。" >&2
  exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[offman] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
