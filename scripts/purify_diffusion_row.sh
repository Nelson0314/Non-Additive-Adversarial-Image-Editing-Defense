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

# 三個路徑可由環境變數覆寫，讓同一支也能量不動點項那一批的防禦圖。
# **影像清單要跟著換**：那一批是三張的篩選批，十張的清單會找不到防禦圖。
SRC="${SRC:-runs/ip2p_mainline}"
OUT="${OUT:-runs/ip2p_fixedpoint/diffusion}"
GAL="${GAL:-runs/gallery_fixedpoint}"
LIST="${LIST:-}"
mkdir -p "$OUT" "$GAL"

[ -f "$DIFFPURE_CKPT" ] || {
  echo "錯誤：找不到 $DIFFPURE_CKPT。缺了它整個算子會被靜默跳過。" >&2; exit 3; }

# 十張，依任務族群切兩片，不用流水號。給了 `LIST` 就改用那份清單，
# 並且**不分片**——篩選批只有三張，切開反而讓每個 process 都要重載一次模型。
A="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
B="task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"
SHARDS="color object"
if [ -n "$LIST" ]; then
  [ -f "$LIST" ] || { echo "錯誤：找不到 $LIST" >&2; exit 2; }
  A=$(tr '
' ' ' < "$LIST"); B=""; SHARDS="color"
fi
N_EXPECT=$(printf '%s
' $A $B | grep -c .)

# `identity` 不可省：它是 retention 的分母，也是「未淨化那一側」的讀數。
PUR="${PUR:-identity gridpure}"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

DEVS=(${1:-})
TAGS=(${2:-floor ours_pg_m ours_ph_q dct_aj85})
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

n_sh=$(printf '%s
' $SHARDS | grep -c .)
n=$(( ${#TAGS[@]} * n_sh ))
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  echo "      每卡 3 個實測會 OOM（docs/OPERATIONS.md）" >&2
  exit 2
fi

# 地板的防禦圖就是原圖，借任何一個已完成的目錄當影像來源即可。
FLOOR_SRC=""
for d in "$SRC"/*/; do
  [ "$(ls -1 "$d"/*__def.png 2>/dev/null | wc -l)" -ge "$N_EXPECT" ] && FLOOR_SRC="${d%/}" && break
done

i=0
for tag in "${TAGS[@]}"; do
  if [ "$tag" = floor ]; then
    run="$FLOOR_SRC"; extra="--floor"
    [ -z "$run" ] && { echo "錯誤：地板需要一個已完成的防禦目錄當影像來源" >&2; exit 3; }
  else
    run="$SRC/$tag"; extra=""
    # **逐張檢查而不是數總數**：十張的目錄拿來跑三張的子集是合法的，
    # 數總數會把它誤判成「防禦圖不齊」。要擋的是「這一張沒有防禦圖」。
    for im in $A $B; do
      ls "$run"/${im}__*__def.png >/dev/null 2>&1 || {
        echo "錯誤：$tag 缺 $im 的防禦圖" >&2; exit 3; }
    done
  fi
  for sh in $SHARDS; do
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
