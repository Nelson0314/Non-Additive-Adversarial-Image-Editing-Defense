#!/usr/bin/env bash
# 分階段訓練的**篩選批**：階段一照現行主線跑完，階段二在它附近做受約束的
# 再最佳化，把多種淨化算子依序輪流餵進前向。
#
# 要回答什麼
# ────────────────────────────────────────────────────────────────────
# `docs/RESULTS.md` 已否決「針對淨化最佳化」：三個變體（固定 JPEG75／課程
# 排程／多算子 EOT）在四個算子上都沒超過基準，而**未淨化強度掉 10–25%**。
# 幾何 EOT 另跑過一次，crop 淨增益升到全表最高（0.1203，首次超過 DCT-Shield
# 的 0.1098），但未淨化位移由 0.6710 掉到 0.5741——失去 0.0969 換回 0.0298，
# **帳差三倍多**。
#
# 兩批的死法是同一個：**沒有任何東西擋住未淨化那一側往下掉**。本批針對這個
# 病灶，改動只有兩處：
#
#   1. **不從零開始。** 階段一是現行主線 `ours_pg_m`（`--radius 2.0`、含加性
#      下限，DISTS 0.1453、blur1 淨增益 0.12099、crop0.1 0.10423，十張）跑完
#      的解；階段二以它為起點、用小步長在附近找一個比較耐洗的鄰居。
#   2. **信賴域。** 每 20 步量一次未淨化的增益 `L(原圖) − L(防禦圖)`，低於
#      階段一終值的 `--stage2-trust` 倍就退回上一個通過的快照並把步長減半；
#      步長掉到初始值的 1/32 以下就停。交出去的一律是最後一個通過檢查的快照。
#
# **這等於把一條已否決的路重開**，使用者已就本批次明確授權。結果不論好壞
# **都不會自動用來推翻或恢復** `RESULTS.md` 上那條裁定——那條是對「從零開始
# 的擴增訓練」下的，本批是另一種東西。
#
# 算子為什麼依序輪替而不是每步隨機
# ────────────────────────────────────────────────────────────────────
# 每步獨立隨機抽在只跑幾百步時短期覆蓋不均：某個算子連著被抽好幾次、另一個
# 一次都沒抽到，方向會被偏掉。`--stage2-order shuffle`（預設）每一輪把清單
# 洗牌再依序走完，覆蓋均勻；不用固定輪替是因為那樣每一輪的最後一個算子恆是
# 「最後說話的那個」，配上 sign 更新容易走成週期性的來回。
#
# 三個工作點與它們各自回答什麼
# ────────────────────────────────────────────────────────────────────
#   s2_tight   保守：400 步、步長 1/5、信賴域 0.95
#   s2_hard    兇：  800 步、步長 1/2、信賴域 0.80
#   s2_null    **歸因對照**：與 s2_hard 逐字相同，只是算子池只有 identity。
#              第二段多跑 800 步本身就可能讓數字變好，**沒有這一格，不管
#              結果是好是壞都歸因不清**。
#
# 兩個點合起來給的是一條取捨曲線（未淨化賠了多少 vs 淨化後賺了多少），不是
# 一個點——幾何 EOT 那次就是只有一個點，回答不了「這筆帳有沒有可能打平」。
#
# 事前判準（跑之前寫下）
# ────────────────────────────────────────────────────────────────────
#   S1  信賴域若一路咬著不放（`stage2_reverts` 每次檢查都退、`stage2_alpha_final`
#       掉到底），結論是「這個工作點附近沒有可行的改善方向」。這是乾淨的
#       負面結果，不是失敗的批次。
#   S2  blur1 與 crop_resize0.1 的扣地板淨增益要**一起**上升，且至少一欄的
#       增幅大於該欄逐圖的標準差。只有一欄動、另一欄掉是在兩欄之間搬錢。
#   S3  增幅必須明顯大於 `s2_null`（純粹多跑步數）的那一份。
#   S4  取捨曲線的斜率（淨增益增幅 ÷ 未淨化位移損失）要大於 1，否則就是重蹈
#       幾何 EOT 那筆帳。
#
# 三張影像怎麼選
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_stage2/images3.txt`。規則是**每個任務型態取字母序第一張**，取
# 色／景／物三型；`task_obj_remove_*` 不入選是因為移除型的指令服從率最低
# （`DECISIONS.md`：移除物件 1/5），三張的批次不該把一格花在最不穩的型態上。
# **這是篩選批，只回答「有沒有抬升的跡象」**；有跡象才擴到十張。
#
# 用法：bash scripts/stage2_screen.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_stage2
mkdir -p "$OUT"
LIST="$OUT/images3.txt"
[ -f "$LIST" ] || { echo "錯誤：找不到 $LIST" >&2; exit 2; }
IMGS=$(tr '\n' ' ' < "$LIST")

# 階段一逐字等於 `scripts/mainline_defense.sh` 的 `ours_pg_m`。**不可以改**：
# 改了就沒有已跑過的十張基準可以對，S2／S3 兩條判準都會失去參照。
BASE="--data data/omniedit150 --loss latent_norm --steps 1000 --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --spectral-floor 0.04 \
--conditions phase_gain --gain-ratio 1.0 --radius 2.0"

OPS="identity blur1 blur2 crop05 crop10 crop15"

POINTS="
s2_tight:--stage2-steps~400~--stage2-step-scale~0.2~--stage2-trust~0.95
s2_hard:--stage2-steps~800~--stage2-step-scale~0.5~--stage2-trust~0.80
s2_null:--stage2-steps~800~--stage2-step-scale~0.5~--stage2-trust~0.80~--stage2-ops~identity
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

# **別人的卡一律不碰。** 指定的卡上只要有別人的 process 就拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  echo "      每卡 3 個實測會 OOM（docs/OPERATIONS.md），不可以硬塞" >&2
  exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  ARGS=$(echo "$extra" | tr '~' ' ')
  # `s2_null` 自己帶 `--stage2-ops identity`，其餘用共用的算子池。
  case "$ARGS" in *--stage2-ops*) POOL="" ;; *) POOL="--stage2-ops $OPS" ;; esac
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $ARGS $POOL \
      --images $IMGS > "$OUT/$tag.log" 2>&1 &
  echo "[stage2] $tag dev=$dev pid=$!  $ARGS $POOL"
done
echo "[stage2] 送出 $i 個（$(date)）"
