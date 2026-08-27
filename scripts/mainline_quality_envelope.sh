#!/usr/bin/env bash
# 兩邊的壓縮品質旗鈕各自取最佳包絡，再逐一攻擊品質比較。
#
# 要回答什麼
# ────────────────────────────────────────────────────────────────────
# 主線頭對頭上 DCT-Shield 一律是 `Q_alg = 0.85`（§6.3 圖 6 的 Y-only 設定），
# 而那個設定的抗 JPEG 保證是**單向**的（補充材料 D.4）——它是為了活過品質
# >= 85 的重壓而設。我們卻拿品質 30 去攻擊它。`Q_alg` 是**防禦方自選**的，
# 對手若知道攻擊方會壓到 30，照論文用法就該把 `Q_alg` 一起調低。不補這一格，
# 「你贏是因為對手把旗鈕設錯」是一句打不掉的質疑。
#
# 對稱地，本方法的 `--deliver-jpeg QD` 是同一個旗鈕，而
# `runs/ip2p_deliver_jpeg/` 的低 QD 族**只跑過防禦、從未跑過抗淨化**。
# 只補對手那一邊就是挑對自己有利的比法，兩邊一起放才是包絡對包絡。
#
# 判準寫在前面（不是看到數字才定）
# ────────────────────────────────────────────────────────────────────
#   1. 看的是**曲線的形狀**不是單一格。低 `Q_alg`／低 `QD` 應該在 jpeg30 那端
#      抬起來、jpeg90 那端塌下去。六條曲線若只是整體平移沒有換形狀，
#      「壓得越狠我方越有優勢」就與旗鈕無關，這條路對雙方都是死的。
#   2. 失真必須拉回等失真再比，用 `scripts/matched_distortion_table.py` 內插。
#      **固定 eps 或固定半徑下比不同條件是錯的**，先前有三個結論這樣出錯過。
#   3. 若對手在等失真下把 jpeg30 追上來，本方法僅存的那一條主主張就沒了。
#
# eps 與半徑怎麼定的（**不是猜的**）
# ────────────────────────────────────────────────────────────────────
# `eps` 的單位是量化後的整數係數，一格換算回像素的幅度等於量化表值，而量化表
# 隨 `Q_alg` 反向縮放（libjpeg：Q=85 的表是 base 的 0.30 倍，Q=30 是 1.67 倍）。
# **同一個 eps 在不同 `Q_alg` 上不是同一個失真預算**，差到 5.6 倍。逐點由
# `scripts/dct_shield_eps_calibration.py` 在本機（不跑 GPU）先量出括弧，
# 該支的飽和代理對三個已知 GPU 點的偏差一致落在 1.17–1.18 倍，故除以 1.176
# 換算成預期真值。`matched_distortion_table.py` 拒絕外插，猜錯就是整批白跑。
#
# **一個由校準本身就看得到、但仍要實測確認的張力**：DCT-Shield §4.2 的抗 JPEG
# 條件要求 |δ| 至少一個量化階（eps >= 1），否則同品質重壓的四捨五入就能把它
# 抹掉。而低 `Q_alg` 的量化階很大——校準外推指出，要在 `Q_alg = 0.50` 保住
# eps >= 1，失真約到 DISTS 0.28；`Q_alg = 0.30` 約 0.41。兩者都遠在失真帶
# （0.1286–0.1447）與本方法最強點（0.1947）之上。也就是**對手要降 `Q_alg`
# 又留在帶內，就得讓 eps 掉到 1 以下，等於放掉它自己的保證**。本批的六個
# 對手點因此全部 `eps < 1`，`DCTShieldSpec` 會自動標 `modified_from_paper`
# 並寫入 CSV 的 `modification_note`，報表上不可略去。
#
# 用法：
#     bash scripts/mainline_quality_envelope.sh "<卡號>" [defense|purify] ["<tag> ..."]
#
# `defense` 產防禦圖與失真讀數；`purify` 讀那些**已存的**防禦圖跑抗淨化。
# 只跑第一步的話 net_gain 那張表上不會有這些條件。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_mainline
PUROUT=runs/ip2p_mainline_purify
GAL=runs/gallery_mainline
mkdir -p "$OUT" "$PUROUT" "$GAL"

IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

# 分片與 `mainline_purify.sh` 逐字相同，跨批可比。
A="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
B="task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

# 抗淨化只留 JPEG 家族：這一批問的是壓縮品質旗鈕，blur 與 crop 三個方法都沒有
# 防禦（`ip2p_mainline/README.md` 第六節），跑了不會改變任何結論而成本翻倍。
# `identity` 不可省，它是 retention 的分母。空白地板已齊（`floor_*.csv`，
# 十張九算子），地板與條件無關，不重跑。
PUR="${PUR:-identity jpeg90 jpeg75 jpeg50 jpeg30}"

# **分隔用 `~` 不是空白。** `POINTS` 是靠 `for p in $POINTS` 做詞彙切分的，
# 這個字串一旦含空白就會被拆成好幾個「工作點」，實際送出去的只剩
# `--out ... --conditions`，argparse 當場報錯（踩過一次）。旗標之間一律用 `~`，
# 由派工那一行的 `tr '~' ' '` 還原。
OURS_COMMON="--conditions~phase~--gain-ratio~0~--loss~latent_norm~--steps~1000~--quantile~0~--freq-weight~jpeg_luma~--freq-weight-power~0.25~--hop~8~--spectral-floor~0.04"

# tag:旗標
#
# 對手側：`Q_alg` 0.50 與 0.30，各三個 eps 夾住 0.0928（ours_ph_q）到
# 0.1947（ours_pg_q20）整段。
# 我方側：交付品質 0.60／0.45／0.35，半徑由 `deliver_quality_calibration.py`
# 定，理由同上——交付自壓把 DISTS 往上推一個與半徑幾乎無關的常數。
POINTS="
dct_aj50_eps0.22:--conditions~dct_shield_y~--q-alg~0.50~--eps~0.22~--dct-steps~1000
dct_aj50_eps0.40:--conditions~dct_shield_y~--q-alg~0.50~--eps~0.40~--dct-steps~1000
dct_aj50_eps0.65:--conditions~dct_shield_y~--q-alg~0.50~--eps~0.65~--dct-steps~1000
dct_aj30_eps0.13:--conditions~dct_shield_y~--q-alg~0.30~--eps~0.13~--dct-steps~1000
dct_aj30_eps0.25:--conditions~dct_shield_y~--q-alg~0.30~--eps~0.25~--dct-steps~1000
dct_aj30_eps0.42:--conditions~dct_shield_y~--q-alg~0.30~--eps~0.42~--dct-steps~1000
ours_ph_qd60:${OURS_COMMON}~--radius~0.9~--deliver-jpeg~0.60
ours_ph_qd45:${OURS_COMMON}~--radius~0.9~--deliver-jpeg~0.45
ours_ph_qd35:${OURS_COMMON}~--radius~0.9~--deliver-jpeg~0.35
"

DEVS=(${1:-})
STAGE="${2:-defense}"
ONLY="${3:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [defense|purify] [\"<tag> ...\"]" >&2; exit 2; }
case "$STAGE" in defense|purify) ;; *)
  echo "用法：$0 \"<卡號>\" [defense|purify] [\"<tag> ...\"]" >&2; exit 2 ;; esac

# **別人的卡一律不碰**，這一道是強制的。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

TAGS=()
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && TAGS+=("$tag"); done
[ ${#TAGS[@]} -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }

# 每卡最多兩個 process（`docs/OPERATIONS.md`）。超過就拒絕，不讓卡號公式繞回去
# 疊加——實測疊到四個就整批 CUDA OOM，而且是跑了十幾分鐘之後才掛。
need=${#TAGS[@]}
[ "$STAGE" = "purify" ] && need=$(( need * 2 ))
if [ "$need" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$need 個 process 需要至少 $(( (need + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  echo "分批送：第三個參數限定 tag。" >&2; exit 2
fi

i=0
if [ "$STAGE" = "defense" ]; then
  for p in $POINTS; do
    IFS=: read -r tag extra <<< "$p"
    selected "$tag" || continue
    dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
        --out "$OUT/$tag" $(echo "$extra" | tr '~' ' ') \
        --images $IMGS --data data/omniedit150 \
        > "$OUT/$tag.log" 2>&1 &
    echo "[envelope/defense] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
  done
else
  # **不自己拼 phase_retention.py 的旗標**：`mainline_purify.sh` 已經帶著
  # 「防禦圖必須 10/10」的守門、地板來源的挑選、以及與既有批次逐字相同的
  # 分片與旗標。在這裡重寫一份，旗標一旦漂移，新舊批次就不可比而且看不出來。
  PUR="$PUR" exec bash scripts/mainline_purify.sh "${DEVS[*]}" "${TAGS[*]}"
fi
wait
