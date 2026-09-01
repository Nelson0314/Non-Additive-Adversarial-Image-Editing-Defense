#!/usr/bin/env bash
# 可學空間包絡（`--floor-envelope`）的防禦端與抗淨化兩段。
#
# 這一支在問什麼
# ────────────────────────────────────────────────────────────────────
# 本方法目前被裁切與模糊洗掉。兩者的機制不同但交集明確：
#
#   模糊  在頻域乘 `exp(-2*pi^2*sigma^2*f^2)`，**與擾動放在畫面哪裡無關**。
#         空間上壓得越窄、頻譜攤得越寬，單位預算下被模糊拿走的**更多**。
#         能穿過 sigma = 2 的只有低頻，在空間上意味著「大而平滑」。
#   裁切  `crop_resize0.1` 是繞中心放大 1.2488x，落在中央 80% 以內的才留下，
#         重取樣本身又是一次低通。所以裁切要的是**靠中心**。
#
# 交集是「靠近中心的、大尺度的、平滑的結構」。本模組原本沒有**可學的空間
# 定位**這個旋鈕：紋理閘會定位但由原圖決定、不可學，`--floor-survival` 已經
# 在頻率上做軟性挑選但空間上均勻。這一批把那個旋鈕加上去。
#
# **總預算不隨包絡改變**（`PhaseResidual.envelope_price` 把價目表縮放回同一個
# 平均值），所以「集中」的意思是「同樣的總量堆到少數位置，那些位置的定價
# 因此被抬高」——也就是「擾動很大但只在一個地方」。集中之後 L2 會上升，
# 那由量測協定的兩個失真軸照實回報，不在構造裡預先修正。
#
# 四個工作點
# ────────────────────────────────────────────────────────────────────
#   env_off        對照：與 `sb_surv` 逐位元同構造，包絡關閉。**從零跑**，
#                  不由 `sb_surv` 續跑——續跑的起點走過 12000 步，與另外
#                  三格的步數預算不同，那會變成第二個變因。
#   env_floor      包絡只乘加法項的價目表（K = 1）。加法項是唯一不受
#                  `|S_b(w)|` 限制、能自由選擇放在哪裡的那一半。
#   env_all        包絡也乘進相位與增益的閘（K = 1）。
#   env_floor_mix  同 `env_floor` 但 K = 3 的軟聯集——單一凸包放不下兩個
#                  主體時的對照。
#
# 收斂判定
# ────────────────────────────────────────────────────────────────────
# `--steps 8000` 只是上限，實際停止由固定抽樣的評估決定。三個數字沿用
# `sbsurv_to_convergence.sh` 已經校準過的那一組，**理由是評估自身的噪聲**：
# 兩張圖的固定評估相對標準差是 0.86% 與 1.49%，先前的 `--min-delta 0.0002`
# （0.02%）是它的 43 與 75 分之一，於是每隔幾次評估就有一次純粹因抽樣而落到
# 歷史最佳以下，`patience` 的計數器被不斷重置、早停永遠不觸發。
#
#   `--eval-draws 16`   噪聲按 1/sqrt(n) 降，預期由 0.9-1.5% 降到 0.6-1.1%。
#   `--min-delta 0.01`  要求改善**大於評估噪聲本身**。低於噪聲的「改善」
#                       不可與抽樣運氣區分，不該重置計數器。
#   `--patience 6`      配 `--eval-every 400` = 連續 2400 步沒有超過 1% 的
#                       改善才停。
#
# **這是收斂判準不是效果判準**（`CLAUDE.md` 的「訓練方法的實驗：不設判準」
# 允許前者、禁止後者）：它只決定什麼時候停，不決定這個方法有沒有效。
# **停在 `max_steps` 的一律照實報成未收斂**，不要含糊帶過。
#
# 用法
# ────────────────────────────────────────────────────────────────────
#     bash scripts/floor_envelope_to_convergence.sh train  "<卡號...>"
#     bash scripts/floor_envelope_to_convergence.sh purify "<卡號...>" [條件...] [nofloor]
#
# 兩段都最多吃 5 張卡。`purify` 只讀已存的防禦圖，不重跑防禦。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

PHASE="${1:-}"
case "$PHASE" in
  train|purify) ;;
  *) echo "用法：$0 train|purify \"<卡號...>\" [條件...] [nofloor]" >&2; exit 2 ;;
esac

DEVS=(${2:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 $PHASE \"<卡號...>\"" >&2; exit 2; }
# 上限 5 張。卡是多人共用的，超過就是在擠別人。
MAX_CARDS=5
if [ ${#DEVS[@]} -gt "$MAX_CARDS" ]; then
  echo "錯誤：最多 $MAX_CARDS 張卡，收到 ${#DEVS[@]} 張。" >&2; exit 2
fi
# **檢查要擋在派工前面。** 指定的卡只要有一張是別人的就拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_floor_envelope
mkdir -p "$OUT"
# 使用者指定：只跑這兩張，省機時。
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"

# `sb_surv` 的設定，四格共用。唯一的變因是包絡那三個旗標。
BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 \
--spectral-floor 0.08 --floor-survival blur12 \
--purify-aware eot_broad --eot-sigmas 0.5 1.0 2.0 3.0 \
--steps 8000 --step-size 0.01 \
--eval-every 400 --eval-draws 16 --patience 6 --min-delta 0.01 \
--save-weights --skip-existing"

TAGS="env_off env_floor env_all env_floor_mix"
flags_for() {
  case "$1" in
    env_off)       echo "" ;;
    env_floor)     echo "--floor-envelope gauss --floor-envelope-scope floor" ;;
    env_all)       echo "--floor-envelope gauss --floor-envelope-scope all" ;;
    env_floor_mix) echo "--floor-envelope gauss --floor-envelope-scope floor --floor-envelope-k 3" ;;
    *) echo "__unknown__" ;;
  esac
}

# 每卡最多 2 個 process（`docs/OPERATIONS.md`：3 個實測整批 OOM，而且是跑了
# 十幾分鐘之後才掛）。超過就拒絕啟動並說要幾張卡，不靜默疊加。
require_slots() {
  local n="$1"
  if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
    echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
    exit 2
  fi
}

i=0
next_dev() { local d=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1)); echo "$d"; }

if [ "$PHASE" = "train" ]; then
  require_slots "$(echo $TAGS | wc -w)"
  for t in $TAGS; do
    extra="$(flags_for "$t")"
    dev="$(next_dev)"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
        --out "$OUT/$t" $BASE $extra --images $IMGS \
        < /dev/null >> "$OUT/$t.log" 2>&1 &
    disown
    echo "[train] $t dev=$dev  上限 8000 步，早停 patience 6 x 400  $extra"
  done
  sleep 25
  echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
  echo
  echo "收的時候先讀 results.csv 的 stop_reason / stopped_at："
  echo "  early_stop 才算收斂；仍是 max_steps 就照實記為未收斂。"
  echo "學出來的包絡在同一份 CSV 的 env_beta / env_fc / env_cy0 / env_cx0 /"
  echo "env_sigma0 幾欄（換算成像素的是 env_py0 / env_px0 / env_sigma_px0）。"
  exit 0
fi

# ---- purify ----
#
# **只讀已存的防禦圖，不重跑防禦。** 空白地板不可省略（`docs/GOAL.md`）：
# 不扣掉它，「淨化後位移較大」無法排除「該算子本來就把編輯推得比較開」
# 這個平庸解釋。**這一批要自己跑地板**，不可沿用別批的——幾何類算子的參照
# 已換成「同一個算子淨化過的原圖」，舊地板在裁切欄是舊參照量的。
#
# `--gallery` 一律開：擋下與否要用眼睛判「重畫」對「劣化」，SigLIP 代理實測
# 會把「人還在、只是被蓋上紋理」標成 blocked。這一批要判的正是「擾動有沒有
# 從主體挪開」，圖比數字更直接。
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"
WANT="${3:-$TAGS}"
WANT_FLOOR=1
[ "${4:-}" = "nofloor" ] && WANT_FLOOR=0

# 只送有防禦圖的條件。缺的一律印出來，不靜默略過。
GOOD=""
for t in $WANT; do
  if ls "$OUT/$t"/*__def.png >/dev/null 2>&1; then GOOD="$GOOD $t"
  else echo "[skip] $OUT/$t 沒有防禦圖" >&2; fi
done
[ -z "$GOOD" ] && { echo "錯誤：沒有任何條件有防禦圖" >&2; exit 2; }
FIRST=$(echo $GOOD | awk '{print $1}')

PUROUT="$OUT/purify"
mkdir -p "$PUROUT"
require_slots "$(( $(echo $GOOD | wc -w) + WANT_FLOOR ))"

launch() {              # $1 輸出標籤  $2 防禦圖來源  $3 額外旗標
  local tag="$1" run="$2" extra="${3:-}"
  local dev="$(next_dev)"
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
      --run "$OUT/$run" $COMMON --images $IMGS $extra \
      --out "$PUROUT/${tag}_all.csv" --gallery "$PUROUT/gallery_${tag}" \
      < /dev/null >> "$PUROUT/${tag}.log" 2>&1 &
  disown
  echo "[purify] $tag dev=$dev $extra"
}

for t in $GOOD; do launch "$t" "$t"; done
# 地板的 `--run` 指到第一個條件的目錄：`--floor` 把**原圖**當防禦圖，來源
# 目錄只用來定位這一批的設定，不會讀到它的防禦圖。
[ "$WANT_FLOOR" -eq 1 ] && launch floor "$FIRST" --floor

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention process"
echo "出表：$PY scripts/retention_table.py --src $PUROUT"
