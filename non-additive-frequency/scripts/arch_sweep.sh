#!/usr/bin/env bash
# 架構各處的旗標掃描。**一輪四個工作點、兩張影像、8000 步，用眼睛判定。**
#
# 共同基準
# ────────────────────────────────────────────────────────────────────
# 除了該輪要掃的那個旗標，其餘與 `ig_d25` 完全相同（image_guidance、
# diffuse_src、radius 2.5、固定步長 0.01、8000 步上限）。每一輪都自帶一個
# 基準線點，因為「改善來自旗標還是來自別的」只有同批的基準線分得出來——
# 2000 步那次就是漏了這件事（`runs/ip2p_pixel_matched/README.md`）。
#
# 判定由助手的視覺做（使用者授權）：防禦圖有沒有波紋、編輯結果是重畫還是
# 劣化。**擋下數不可用 SigLIP 代理**——它把「人還在、只是蓋了紋理」也標成
# blocked（實測 `ih_l15`）。
#
# 目前為止唯一兩軸都沒變差的旗標是 `--theta-budget 0.10`（`ip2p_theta_budget`：
# 對同批基準線 DISTS 低 7.8%、PSNR 高 1.45 dB、SSIM 高 0.034，位移略高，
# 編輯仍是重畫），所以第三輪之後的掃描都把它接上當底。
#
# 用法：ROUND=<名稱> bash scripts/arch_sweep.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：ROUND=<名稱> $0 \"<四個卡號>\"" >&2; exit 2; }
ROUND="${ROUND:-}"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"
TB="--theta-budget~0.10"

case "$ROUND" in
  tbfine)
    # 把唯一有效的那個旗標掃細。0.10 兩側各兩點，看拐點在哪。
    OUT=runs/arch_theta_fine
    POINTS="tf_006:--theta-budget~0.06 tf_008:--theta-budget~0.08 \
tf_014:--theta-budget~0.14 tf_020:--theta-budget~0.20" ;;
  power)
    # 知覺定價的指數。`docs/PENDING.md` 記著 0.25→0.35 從未在等失真下掃過。
    OUT=runs/arch_pricing_power
    POINTS="pw_025:$TB~--freq-weight-power~0.25 pw_035:$TB~--freq-weight-power~0.35 \
pw_050:$TB~--freq-weight-power~0.50 pw_010:$TB~--freq-weight-power~0.10" ;;
  rmin)
    # 頻帶下界。FND-042 已記「拉高一致更好」，保留 0.12 的理由（既有批次的
    # 基準）早已不成立，但在 image_guidance 這條損失上從未掃過。
    OUT=runs/arch_band_lower
    POINTS="rm_012:$TB~--r-min~0.12 rm_020:$TB~--r-min~0.20 \
rm_028:$TB~--r-min~0.28 rm_036:$TB~--r-min~0.36" ;;
  edge)
    # 紋理閘裡壓制邊緣那個因子的指數。從未掃過。0 = 不壓邊緣。
    OUT=runs/arch_edge_power
    POINTS="ed_000:$TB~--gate-edge-power~0 ed_100:$TB~--gate-edge-power~1.0 \
ed_200:$TB~--gate-edge-power~2.0 ed_300:$TB~--gate-edge-power~3.0" ;;
  chan)
    # 動哪些通道、以及要不要保留乘性增益。純相位在 PENDING 上兩個錨點都較便宜。
    OUT=runs/arch_channels
    POINTS="ch_rgb_g1:$TB ch_y_g1:$TB~--phase-channels~y \
ch_rgb_g0:$TB~--gain-ratio~0 ch_y_g0:$TB~--phase-channels~y~--gain-ratio~0" ;;
  consmatch)
    # 一致性懲罰**加到等失真**。兩張的讀數（`runs/ip2p_consistency`）：
    #   w 0     DISTS 0.1643  PSNR 21.87  位移 0.6439   位移/DISTS 3.92
    #   w 0.03  DISTS 0.1389  PSNR 25.09  位移 0.5838   4.20
    #   w 0.10  DISTS 0.1096  PSNR 26.46  位移 0.5672   5.18
    #   w 0.30  DISTS 0.0891  PSNR 27.78  位移 0.5438   **6.10**
    # 失真單調降 46%、PSNR 升 5.9 dB，而位移只掉 16%。**每單位失真換到的
    # 位移一路往上**，而且 w=0.30 的 1.56 倍遠超出專案量過的跨方法變異
    # （CV 0.108）——今天所有旗標裡只有這一個是這個形狀。
    #
    # 但那不是等失真的比較。本輪把兩個最好的權重加半徑，跨過基準線的
    # DISTS 0.1643，**不必外插**；另加一個更高的權重看單調性有沒有到頭。
    OUT=runs/arch_cons_matched
    POINTS="cm_w10r35:--consistency-weight~0.10~--radius~3.5 cm_w30r45:--consistency-weight~0.30~--radius~4.5 cm_w30r60:--consistency-weight~0.30~--radius~6.0 cm_w100r45:--consistency-weight~1.00~--radius~4.5" ;;
  consanneal)
    # 一致性權重**退火**。固定權重會把失真封頂：`runs/arch_cons_matched` 實測
    # w=0.30 的半徑由 2.5 拉到 6.0（2.4 倍），DISTS 只由 0.0835 走到 0.1145，
    # 到不了基準線的 0.153；而重畫與否看起來由**失真水準**決定——0.15 附近
    # 重畫、0.11 附近劣化。也就是懲罰項把圖變好看的方式是不去它有效的地方。
    #
    # 退火問的是「懲罰是不是只在早期需要」：先用它把解推進可實現的盆地，再把
    # 權重線性降到零，讓失真長回 0.15 而形狀留下來。
    # `an_none` 是同批基準線（權重固定、不退火），沒有它分不出改善來自退火
    # 還是別的。**歸零位置的三個值沒有出處，是本專案指定的。**
    OUT=runs/arch_cons_anneal
    POINTS="an_none:--consistency-weight~0.30 an_d25:--consistency-weight~0.30~--consistency-decay~0.25 an_d50:--consistency-weight~0.30~--consistency-decay~0.50 an_d75:--consistency-weight~0.30~--consistency-decay~0.75" ;;
  *) echo "錯誤：未知的 ROUND=$ROUND" >&2; exit 2 ;;
esac

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
mkdir -p "$OUT"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--r-min 0.12 --gate-edge-power 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[$ROUND] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"
