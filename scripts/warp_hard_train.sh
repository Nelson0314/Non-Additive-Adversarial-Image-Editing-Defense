#!/usr/bin/env bash
# 位移場：**硬訓練到收斂**。探索性質，不是要馬上當成果。
#
# 要回答什麼
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_warp/DIAGNOSIS.md` 已判定：**預算從頭到尾就夠**（L∞ 球在 4 px 的
# 天花板已是 DISTS 0.1402，落在失真帶內，而跑過的最小半徑正是 4 px），
# 最佳化只用掉天花板的 7–15%，而且走出來的東西與**擲硬幣無法區分**
# （與無偏隨機遊走 `α√N` 的比值 0.40–0.84）。
#
# 病灶指名了兩個，本批把兩個都拆掉：
#
#   1. **起點就在病灶上。** `WarpParam.reset` 把場設成全零，而
#      `step_probe_latent_norm.csv` 量到 `latent_norm` 在零位移處有一個
#      **帶折點的局部極小**：梯度完全正常但每走一步損失都上升，sign PGD 於是
#      在 0 與 ±α 之間形成週期 2 的振盪。→ `--warp-init-std`
#   2. **步長綁在半徑上**（`α = radius/(steps·saturate_at)`），於是「放寬預算」
#      同時「放大步長」，放寬預算不等於走得更遠。→ `--step-size`
#
# 第三個變因是**更新規則**：sign PGD 在帶折點上必然振盪，那是更新規則的性質
# 不是損失的性質。→ `--update adam`
# **`docs/GOAL.md` 把「Adam 更新規則」列在更早期已否決的方向裡**，但那批證據
# 已刪除、且是在**相位**參數化上做的，與位移場不是同一件事；使用者已就本探索
# 批次明確授權。**本批的結果不可用來推翻或恢復相位臂上的那個否決。**
#
# 第四個變因是**粗網格的細緻度**。`budget_utilization.csv` 量到
# `coherent`（整張平移，最平滑的場）放到 24 px 的天花板只有 DISTS 0.0982，
# **仍在失真帶之外**——失真來自場的**粗糙度**不是幅度。grid 由 16 加到 64
# 直接提高可達的粗糙度，這是光流那條線最直接的移植。
#
# **不限制對畫面的破壞**：`--radius 64` 實質上不設限（4 px 就夠到帶內了）。
# 先看走不走得動，等失真對齊是下一步的事。
#
# 事前寫下的判準
# ────────────────────────────────────────────────────────────────────
#   W0  每一格都要一併看**場的粗糙度**。平滑場在 24 px 上的天花板只有 0.0982，
#       構造上就到不了帶內；某一格走不到帶內而它的場是平滑的，結論是「收斂到
#       平滑解」，不是「預算不夠」也不是「步數不夠」。
#   W1  任一格的最終 DISTS 構不到 0.1286 → 不規定預算也走不到失真帶。
#   W2  走得到，但等失真下 `edit_lpips` 沒有超過同批 `warp_rand` 的 1.2 倍
#       → FND-004「與同失真隨機無法區分」在修好最佳化之後**仍然成立**。
#   W4  損失軌跡在最後 500 步的相對變化 > 1% 者一律標「未收斂」，
#       **不得拿它的數字下任何結論**——「跑到收斂」是本批的前提不是假設。
#
# 用法：bash scripts/warp_hard_train.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_warp_hard
mkdir -p "$OUT"
# **十張，不是十三張**（`docs/DECISIONS.md`：主線影像已定案為十張）。
# 因此**不可以**與 `matched_geometry.csv` 的十三張讀數並列——本批自己帶
# `warp_rand` 的同批對照就是為了這個。
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

COMMON="--data data/omniedit150 --steps 4000 --radius 64"

# tag:旗標
#
# 兩個 `warp_rand` 不最佳化，半徑由 `budget_utilization.csv` 的 gauss 曲線挑：
# amp 4 px → DISTS 0.138（帶內）、8 px → 0.229。兩點夾住失真帶，供等失真內插。
POINTS="
w_sign_zero:--conditions~warp~--loss~latent_norm~--update~sign~--step-size~0.016~--warp-init-std~0~--warp-grid~16
w_sign_rand:--conditions~warp~--loss~latent_norm~--update~sign~--step-size~0.016~--warp-init-std~1.5~--warp-grid~16
w_adam_zero:--conditions~warp~--loss~latent_norm~--update~adam~--step-size~0.02~--warp-init-std~0~--warp-grid~16
w_adam_rand:--conditions~warp~--loss~latent_norm~--update~adam~--step-size~0.02~--warp-init-std~1.5~--warp-grid~16
w_adam_g64:--conditions~warp~--loss~latent_norm~--update~adam~--step-size~0.02~--warp-init-std~1.5~--warp-grid~64
w_adam_enc:--conditions~warp~--loss~encoder_target~--update~adam~--step-size~0.02~--warp-init-std~1.5~--warp-grid~16
w_rand_r4:--conditions~warp_rand~--radius~4~--warp-grid~16~--steps~1
w_rand_r8:--conditions~warp_rand~--radius~8~--warp-grid~16~--steps~1
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

# **別人的卡一律不碰**，這一道是強制的。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
# 每卡最多兩個 process（`docs/OPERATIONS.md`）。實測疊到四個會整批 CUDA OOM，
# 而且是跑了十幾分鐘之後才掛，所以這一道要擋在派工前面。
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $COMMON $(echo "$extra" | tr '~' ' ') \
      --images $IMGS > "$OUT/$tag.log" 2>&1 &
  echo "[warp-hard] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
wait
