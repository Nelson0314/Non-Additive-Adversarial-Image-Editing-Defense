#!/usr/bin/env bash
# 不動點框架第五節第二步：把「交付到不動點集合」推廣到**擴散淨化**。篩選批。
#
# 為什麼現在做
# ────────────────────────────────────────────────────────────────────
# 框架的協定寫著「只有 D1 通過才做」。D1 已通過：量化交付相對不交付在 jpeg30
# 上是 2.19 倍、在 gridpure 上是 **0.70 倍**——優勢反轉，正如框架所預測
# （JPEG 的量化格點不是擴散淨化器的不動點集合）。
#
# 這一批做的事
# ────────────────────────────────────────────────────────────────────
# 在防禦損失上加一項，要求防禦圖在**攻擊方自己的擴散先驗**底下是高似然的
# （latent 空間的去噪分數匹配，文字與影像兩個條件都丟掉）：
#
#     L_fix(x) = E_{t,eps} || eps_theta( sqrt(a_t) E(x) + sqrt(1-a_t) eps, t ) - eps ||^2
#
# 直覺是：**淨化器找不到東西可以移除，因為在模型眼裡這張圖已經乾淨。**
#
# **先驗取攻擊方的編輯模型，不取評測用的那個淨化器。** 白盒假設給的是編輯模型，
# 不是「攻擊方會挑哪個淨化器」；針對評測算子最佳化就是已否決的 co-adapt。
# 這也讓結果變成**跨模型**的證據：用 SD 1.5 的先驗做出來的圖，若在
# guided-diffusion 建的 GrIDPure 上仍有效，那是轉移性不是過擬合。
#
# **與最接近的前例方向相反**：AntiPure（ICCV 2025）是**破壞**淨化器，本項是
# **迎合**淨化器，兩者損失的符號是反的。related work 要寫這一段。
#
# 三個點
# ────────────────────────────────────────────────────────────────────
#   fp_w1        主線 ＋ 不動點項，權重 1
#   fp_w4        主線 ＋ 不動點項，權重 4（項已正規化，起點恰為 1，故權重可讀）
#   fp_only      **判準 F3 的歸因對照**：只留不動點項、拿掉對抗項。
#                沒有這一格，分不出改善來自「迎合淨化器」還是「失真型態變了」。
#
# 判準（跑之前寫下，見 `runs/fixedpoint_framework/README.md` §5.2）
# ────────────────────────────────────────────────────────────────────
#   F1  等失真下 gridpure 的淨增益要高於現行主線（三張子集：由
#       `runs/ip2p_fixedpoint/diffusion/` 取同樣三張重算，不可用十張平均）。
#   F2  未淨化那一側的損失不得超過量化交付那一筆（等失真 −21%）。超過就代表
#       這一項與防禦強度互斥。
#   F3  `fp_only` 若拿到與 `fp_w*` 相近的 gridpure 淨增益，改善就不是對抗項
#       買到的，主張要跟著改寫。
#
# **三張篩選批**：不動點項每步多一次 UNet 前向與反傳（latent 空間 64²），
# 單點成本約是主線的一點五到兩倍。先回答「有沒有跡象」再擴到十張。
#
# 用法：bash scripts/fixedpoint_manifold_screen.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_fixedpoint/manifold
mkdir -p "$OUT"
LIST=runs/ip2p_stage2/images3.txt
[ -f "$LIST" ] || { echo "錯誤：找不到 $LIST" >&2; exit 2; }
IMGS=$(tr '\n' ' ' < "$LIST")

# 階段一逐字等於 `mainline_defense.sh` 的 `ours_pg_m`——**唯一的變因是不動點項**。
BASE="--data data/omniedit150 --loss latent_norm --steps 1000 --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --spectral-floor 0.04 \
--conditions phase_gain --gain-ratio 1.0 --radius 2.0 --manifold-t 100"

# `fp_w1`／`fp_w4` 用的是 `raw` 相加，而**主損失在乾淨影像上約 70–80、不動點項
# 已正規化成 1**，所以那兩個權重實際只佔約 1/70 與 4/70——實測它們與階段一
# 幾乎逐格相同（DISTS 0.1246／0.1203 對 0.1267，位移 0.7166／0.7132 對 0.7159）。
# `fp_b*` 改用 `--manifold-balance normalised`：兩項都由 1 起步，**權重 1 才是
# 等權**，這才是真正在掃「迎合淨化器」與「破壞編輯」之間的取捨。
POINTS="
fp_w1:--manifold-weight~1.0
fp_w4:--manifold-weight~4.0
fp_only:--manifold-only
fp_b025:--manifold-weight~0.25~--manifold-balance~normalised
fp_b1:--manifold-weight~1.0~--manifold-balance~normalised
fp_b4:--manifold-weight~4.0~--manifold-balance~normalised
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
# 不動點項多帶一份 UNet 的活化，記憶體高於一般工作點，**每卡只放一個**。
if [ "$n" -gt "${#DEVS[@]}" ]; then
  echo "錯誤：不動點項每卡只放一個，$n 個點需要 $n 張卡，只給了 ${#DEVS[@]} 張" >&2
  exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$i]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS > "$OUT/$tag.log" 2>&1 &
  echo "[manifold] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
echo "[manifold] 送出 $i 個（$(date)）"
