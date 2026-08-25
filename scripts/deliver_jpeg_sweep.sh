#!/usr/bin/env bash
# 交付自壓：把 JPEG 量化放進最佳化迴圈，**並且交付壓縮後的圖**。
#
# 要回答什麼
# ────────────────────────────────────────────────────────────────────
# DCT-Shield 抗 JPEG 的全部來源是它的擾動 δ 直接加在**量化後的整數係數**上，
# 於是攻擊方以相同或更高品質重壓時四捨五入不會把它推走。本方法的擾動是連續
# 值，一被重新量化就散掉。假說：交付 `jpeg_roundtrip(x_def, QD)` 之後我們的
# 擾動也落在量化格點上，jpeg75 那一格應該追得上來。
#
# 與已否決項目的區別（**只有一個，但那個差別是關鍵**）
# ────────────────────────────────────────────────────────────────────
# `RESULTS.md` 否決過「針對淨化最佳化沒有改善抗淨化」——`--purify-aware` 的
# 三個變體（fixed75／curriculum／多算子 EOT）把可微分 JPEG 放進 PGD 前向，
# 但**交付的是未壓縮的圖**，擾動一離開迴圈就不在量化格點上，等於白做。
#
#   purify-aware fixed75（已否決）   迴圈看到 JPEG ✓   交付未壓縮的圖
#   DCT-Shield（對手）               迴圈看到 JPEG ✓   交付壓縮的圖
#   本批 --deliver-jpeg              迴圈看到 JPEG ✓   交付壓縮的圖
#
# 判讀規則（跑之前就寫下，不是看到數字才定）
# ────────────────────────────────────────────────────────────────────
#   1. 擾動保留率（CSV 的 `deliver_retention`）**沒有明顯高於 0.22** 就是死的
#      ——0.22 是同一個 QD 下隨機擾動的保留率，代表最佳化沒有學到格點。
#   2. jpeg75 淨增益要追平或超過 `y_q85_e10` 的 0.4324。
#   3. **jpeg30 要一起看**：我們現在贏那一格（0.1463 對 0.0531），自壓到 QD
#      之後攻擊方壓到 30 會穿過交付品質，那個優勢可能被吃掉。
#   4. 失真要落在帶內（DISTS 0.1286–0.1447）才可比；出界用
#      `scripts/matched_distortion_table.py` 內插。
#
# 用法
# ────────────────────────────────────────────────────────────────────
#     bash scripts/deliver_jpeg_sweep.sh "<卡號>" defense
#     bash scripts/deliver_jpeg_sweep.sh "<卡號>" purify
#     bash scripts/deliver_jpeg_sweep.sh "<卡號>" defense "qd75_r15 qd85_r15"
#
# 第三個參數限定要送哪幾個 tag，補送用。
#
# `defense` 產防禦圖與失真讀數；`purify` 讀那些**已存的**防禦圖跑抗淨化。
# 只跑第一步的話 net_gain 那張表上不會有這兩個條件。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
# 不 source ~/env.sh：它最後一行會把工作目錄換到舊的 ~/WACV（坑一）。
# DIFFPURE_CKPT 只寫在那裡，少了它 gridpure／fdpure 會被判為「相依不齊」而
# **靜默跳過**。本批的算子清單不含 gridpure，仍然設好以免將來補跑時忘記。
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_deliver_jpeg
mkdir -p "$OUT"

COLOR="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
SCENE="task_env_weather_112463 task_env_weather_246440 task_env_weather_63722"
OBJECT="task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"
ALL="$COLOR $SCENE $OBJECT"

# 條件與旗標照現行主線 `phase_gain` 與 `runs/ip2p_axis_necessity/b_pg_r20`／
# `b_pg_r15`：半徑取同樣的 2.0 與 1.5，其餘旗標逐字相同，本批與那兩點因此
# **只差 `--deliver-jpeg` 一項**，未交付的對照組不必另跑。
COMMON="--data data/omniedit150 --loss latent_norm --steps 1000 --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 \
--conditions phase_gain --spectral-floor 0.04 --gain-ratio 1.0"

# tag:交付品質:半徑
#
# **每個品質要兩個半徑**，理由是判讀規則第 4 條：交付自壓本身要付一筆失真
# （實測約 +10% DISTS），半徑 2.0 的未交付點是 DISTS 0.1447，正好是失真帶的
# 上緣，加上那一筆就出界了。出界的點只能靠
# `scripts/matched_distortion_table.py` 內插回帶內，而內插至少要兩個強度點。
# 實測的代價比預估大得多，而且它是**加性的**而不是比例的：逐圖看，交付把
# DISTS 往上推一個幾乎與半徑無關的常數（QD 0.85 約 +0.053、QD 0.75 約
# +0.060），因為 JPEG 自己對乾淨影像的那一筆重建誤差不隨擾動強度縮放。
# 後果是半徑 1.5 與 2.0 兩點**都落在失真帶之上**（帶是 0.1286–0.1447），
# 故補半徑 0.9 那一組：未交付是 0.0797（`b_pg_r09`），加上那個常數之後
# 落在帶內，三點因此夾住了整段帶。
# **固定強度下比不同條件是錯的**，先前有三個結論這樣出錯過。
POINTS="
qd85_r20:0.85:2.0
qd75_r20:0.75:2.0
qd85_r15:0.85:1.5
qd75_r15:0.75:1.5
qd85_r09:0.85:0.9
qd75_r09:0.75:0.9
"

# 抗淨化的算子。**不含 gridpure**（那一欄另補，它一格要吃約 82% 的機時）。
PUR="identity blur1 jpeg75 jpeg30 crop_resize0.1 jpeg_then_resize75 adverse_cleaner"
# 空白地板不重跑：13/13 已經齊在 runs/ip2p_purify_headtohead/floor_*.csv，
# 而地板與條件無關（防禦圖就是原圖）。出表時 retention_table.py 會遞迴走進
# 兩個目錄逐圖相減。

DEVS=(${1:-})
STAGE="${2:-defense}"
# 第三個參數：只送這幾個 tag（空白分隔）。**補送用**——卡是一張一張空出來的，
# 整批重送會把已經在跑或已經跑完的點再跑一次，而 `write_csv` 會把那一點既有的
# 列整個蓋掉，中途再斷一次就比現在更少。
ONLY="${3:-}"
if [ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ]; then
  echo "用法：$0 \"<卡號>\" [defense|purify] [\"<tag> ...\"]" >&2; exit 2
fi
case "$STAGE" in defense|purify) ;; *)
  echo "用法：$0 \"<卡號>\" [defense|purify]" >&2; exit 2 ;;
esac

# 每卡最多兩個 process（`docs/OPERATIONS.md`）。超過就拒絕，不可讓卡號公式
# 繞回去疊加——實測疊到四個就整批 CUDA OOM，而且是跑了十幾分鐘之後才掛。
require_slots() {
  local n_points="$1" n_devs="$2"
  if [ "$n_points" -gt $(( n_devs * 2 )) ]; then
    echo "錯誤：$n_points 個 process 需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
    echo "      只給了 $n_devs 張。每卡最多 2 個 process。" >&2
    exit 2
  fi
}

# **別人的卡一律不碰**。這一道是強制的，不是提醒——列印了卻不擋等於沒擋。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() {   # $1 tag
  [ -z "$ONLY" ] && return 0
  for t in $ONLY; do [ "$t" = "$1" ] && return 0; done
  return 1
}

n_points=0
for p in $POINTS; do
  IFS=: read -r tag _ _ <<< "$p"
  selected "$tag" && n_points=$(( n_points + 1 ))
done
if [ "$n_points" -eq 0 ]; then
  echo "錯誤：--only 選到 0 個 tag（收到「$ONLY」）" >&2; exit 2
fi
require_slots "$n_points" "${#DEVS[@]}"

i=0
for p in $POINTS; do
  IFS=: read -r tag qd rad <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  if [ "$STAGE" = defense ]; then
    CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
        --out "$OUT/$tag" --deliver-jpeg "$qd" --radius "$rad" \
        --images $ALL $COMMON \
        > "$OUT/$tag.log" 2>&1 &
    echo "[deliver] $tag QD=$qd radius=$rad dev=$dev pid=$!"
  else
    if [ ! -f "$OUT/$tag/results.csv" ]; then
      echo "[deliver] 跳過 $tag：$OUT/$tag/results.csv 不存在，先跑 defense" >&2
      continue
    fi
    # 只有一張卡時分片得**在同一個 process 裡依序跑**，不能三個一起送。
    # 分片名只能是 color / scene / object——`retention_table.py` 的 `tag_of()`
    # 由檔名還原條件標籤，別的名字會拋錯。
    CUDA_VISIBLE_DEVICES="$dev" nohup bash -c "
      for shard in color scene object; do
        case \$shard in
          color)  imgs='$COLOR' ;;
          scene)  imgs='$SCENE' ;;
          object) imgs='$OBJECT' ;;
        esac
        '$PY' scripts/phase_retention.py --run '$OUT/$tag' --images \$imgs \
            --data data/omniedit150 --attacker ip2p --seeds 3 \
            --purifiers $PUR --out '$OUT/${tag}_'\$shard'.csv'
      done" > "$OUT/${tag}_purify.log" 2>&1 &
    echo "[deliver] $tag 抗淨化（三片依序）dev=$dev pid=$!"
  fi
done
echo "[deliver] $STAGE 全部送出（$(date)），共 $i 個 process"
