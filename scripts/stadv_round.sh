#!/usr/bin/env bash
# stAdv（Xiao et al., ICLR 2018, arXiv:1801.02612）的移植：逐像素稠密流場 ＋
# 根號在內的流場總變差 ＋ L-BFGS。
#
# 為什麼補這一批
# ────────────────────────────────────────────────────────────────────
# `WarpParam` 的 docstring 明寫「本類別**不是** stAdv 的移植，報表上不可寫成
# stAdv 的結果」——本專案用的是 16×16 粗網格上採樣、沒有 TV 項、sign PGD。
# 所以位移場那一族被否決時，**我們手上沒有 stAdv 的數字**，那個否決在論文上
# 站的是自製構造。這一批把它補起來。
#
# 逐行照原文的四項，與必須標明的偏離
# ────────────────────────────────────────────────────────────────────
# 照抄：稠密流場（`--warp-grid 512`，上採樣退化成恆等）、雙線性四鄰居取樣、
# `L_flow = Σ_p Σ_{q∈N(p)} √(‖Δu^p−Δu^q‖²+‖Δv^p−Δv^q‖²)`（根號在鄰居和的
# 裡面）、L-BFGS ＋ line search。
#
# 偏離（全部進 CSV 欄位，見 `src/defense/stadv_flow.py` 的 docstring）：
#   - **`L_adv` 不是原文的那一個。** 原文是分類器 logits 上的 Carlini–Wagner
#     式 `max(max_{i≠t} g_i − g_t, κ)`，κ=0。本專案的攻擊方是擴散編輯模型，
#     沒有 logits，該式在本任務上沒有定義。此處用 `latent_norm`。
#   - **根號內的 eps。** 原文沒有。`f ≡ 0` 時每一項都是 `√0`，梯度是 0/0
#     （實測會給 NaN），而位移場的預設起點正是全零。取 1e-6，成為 CSV 欄位。
#   - **鄰域 `N(p)`。** 原文只寫 `q∈N(p)`。取 `four`（四鄰域），最貼近字面。
#   - **line search 用 strong Wolfe。** 原文寫 backtracking，PyTorch 只提供前者。
#   - **τ 的值。** 原文的 0.05 是在 CW 損失上網格搜尋得到，量級與本專案的損失
#     不同，故不可照抄一個數，掃四點。`sa_t000`（τ=0）把正則項單獨隔出來。
#
# 步數不寫死
# ────────────────────────────────────────────────────────────────────
# 原文未載明迭代次數。`--steps` 只是上限，實際停止由 `--eval-every` ＋
# `--patience` 的收斂判定決定；`--update lbfgs` 缺這兩個旗標會被守門拒絕。
# 停在第幾步與原因記在 `stopped_at`／`stop_reason`，停在上限的照實報為未收斂。
#
# 另外兩個病灶沿用 `runs/ip2p_warp_hard/` 已拆掉的作法：`--warp-init-std 1.5`
# 讓起點離開 `latent_norm` 在零位移處的帶折局部極小，`--radius 64` 讓預算實質
# 不設限（`runs/ip2p_warp/DIAGNOSIS.md` 證明 4 px 就夠到失真帶）。
#
# 用法：bash scripts/stadv_round.sh "<兩個以上卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 2 ] && { echo "用法：$0 \"<兩個以上卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
grep -q -- "--flow-tau" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 --flow-tau，先 git pull" >&2; exit 2; }

OUT=runs/ip2p_stadv
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"

BASE="--data data/omniedit150 --conditions warp --warp-grid 512 \
--loss latent_norm --update lbfgs --radius 64 --warp-init-std 1.5 \
--steps 2000 --step-size 0.02 \
--eval-every 100 --patience 10 --min-delta 0.0002 \
--save-weights --skip-existing"
FLOW="--flow-eps~1e-6~--flow-neighbourhood~four"

POINTS="
sa_t000:--flow-tau~0
sa_t001:--flow-tau~0.001~$FLOW
sa_t010:--flow-tau~0.01~$FLOW
sa_t050:--flow-tau~0.05~$FLOW
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
  echo "[stadv] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
