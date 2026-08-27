#!/usr/bin/env bash
# 不動點框架的第四列：**擴散淨化**。只讀已存的防禦圖，不重跑防禦。
#
# 框架與這一批要檢定什麼
# ────────────────────────────────────────────────────────────────────
# 假說（`docs/reference/SURVEY_FRONTIER.md` §4）：一個防護擾動能不能撐過某個
# 淨化算子，取決於交付出去的影像能不能落在**那個算子的**不動點集合上。四個
# 算子的集合結構完全不同：
#
#   JPEG(Q)      近似冪等的投影      不動點＝量化格點          可行且有效
#   高斯模糊      壓縮映射，不冪等    退化成低頻                結構性不可行
#   裁切縮放      相似變換            log-periodic 的場         寬度為零的窄峰
#   擴散淨化      往資料流形投影      該模型的高似然集合        **本批**
#
# **可證偽的預測（跑之前寫下）**
# ────────────────────────────────────────────────────────────────────
#   D1  量化交付（`ours_ph_q`）相對不交付（`ours_pg_m`）在 **gridpure 上不應該
#       有優勢**——JPEG 的量化格點不是擴散淨化器的不動點集合。它在 jpeg30 上
#       的優勢是 2.19 倍（0.1072 對 0.0489，扣地板淨增益）。**若 gridpure 上
#       也出現同一量級的優勢，框架的「不動點集合是算子專屬的」這條就要修正。**
#   D2  空白地板必須跑。擴散淨化自己就會把編輯推開，不扣掉它，任何「保留率高」
#       都無法排除平庸解釋（`GOAL.md`）。
#   D3  比較一律在等失真下讀：`ours_ph_q` 是 DISTS 0.0928、`ours_pg_m` 是
#       0.1453、`dct_aj85` 是 0.1118。**這一批不是等失真的**，所以它只能回答
#       D1 那個「有沒有優勢」的定性問題，不能拿來排名。
#
# GrIDPure 的超參數**論文正文未載，是本專案指定的**（t=10、gamma=0.1、
# iters=10），由 `scripts/phase_retention.py` 統一給，不在這裡覆寫。
# `DIFFPURE_CKPT` 必須明給：少了它 `Purifier.available` 會判定相依不齊而
# **靜默跳過**整個算子。
#
# 用法：bash scripts/purify_diffusion_row.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

SRC=runs/ip2p_mainline
OUT=runs/ip2p_fixedpoint/diffusion
GAL=runs/gallery_fixedpoint
mkdir -p "$OUT" "$GAL"

[ -f "$DIFFPURE_CKPT" ] || {
  echo "錯誤：找不到 $DIFFPURE_CKPT。缺了它整個算子會被靜默跳過。" >&2; exit 3; }

# 十張，依任務族群切兩片，不用流水號。
A="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
B="task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

# `identity` 不可省：它是 retention 的分母，也是「未淨化那一側」的讀數。
PUR="${PUR:-identity gridpure}"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

DEVS=(${1:-})
TAGS=(${2:-floor ours_pg_m ours_ph_q dct_aj85})
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

n=$(( ${#TAGS[@]} * 2 ))
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  echo "      每卡 3 個實測會 OOM（docs/OPERATIONS.md）" >&2
  exit 2
fi

# 地板的防禦圖就是原圖，借任何一個已完成的目錄當影像來源即可。
FLOOR_SRC=""
for d in "$SRC"/*/; do
  [ "$(ls -1 "$d"/*__def.png 2>/dev/null | wc -l)" -eq 10 ] && FLOOR_SRC="${d%/}" && break
done

i=0
for tag in "${TAGS[@]}"; do
  if [ "$tag" = floor ]; then
    run="$FLOOR_SRC"; extra="--floor"
    [ -z "$run" ] && { echo "錯誤：地板需要一個已完成的防禦目錄當影像來源" >&2; exit 3; }
  else
    run="$SRC/$tag"; extra=""
    n_def=$(ls -1 "$run"/*__def.png 2>/dev/null | wc -l)
    [ "$n_def" -ne 10 ] && { echo "錯誤：$tag 只有 $n_def/10 張防禦圖" >&2; exit 3; }
  fi
  for sh in color object; do
    case $sh in color) IM="$A" ;; object) IM="$B" ;; esac
    dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
        --run "$run" --images $IM $COMMON $extra --gallery "$GAL/$tag" \
        --out "$OUT/${tag}_${sh}.csv" > "$OUT/${tag}_${sh}.log" 2>&1 &
    echo "[fixedpoint] $tag/$sh run=$run dev=$dev pid=$!"
  done
done
echo "[fixedpoint] 送出 $i 個（$(date)）"
