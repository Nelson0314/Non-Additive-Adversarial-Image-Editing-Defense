#!/usr/bin/env bash
# WaNet 式三元對照：位移場的效果是幾何還是內插 artifact，最佳化又買到了什麼。
#
# 要判的是 `RESULTS.md` 的 FND-004（位移場與同失真隨機對照無法區分）——
# 那是這一族方法的通病，還是舊實作的問題。兩者的下一步完全不同，而現有記錄
# 只剩一行結論、證據已刪除。
#
# 機制上的懷疑來自 WaNet（ICLR 2021, arXiv:2102.10369）：沒有被專門訓練去
# 區分位移場的網路，反應的不是位移場的身份，而是**重取樣內插產生的像素級
# artifact**；WaNet 必須額外加入 noise mode（隨機 warp → 標正確類別）才逼得
# 網路學會區分特定的場，而 IP2P 從未受過這種訓練。
#
# 三格與各自回答的問題：
#
#   opt   warp            最佳化的位移場
#   rand  warp_rand       同半徑的隨機場，不最佳化
#   trip  warp_roundtrip  與 rand **同一個**隨機場，先 f 再 −f（往返）
#
#   rand vs trip：幾何本身有沒有貢獻，還是效果全來自內插 artifact？
#   opt  vs rand：最佳化有沒有買到東西？（這一格就是 FND-004）
#
# 半徑的單位是**最大位移像素數**，強度旗鈕是 `--radius`。三格的「半徑 → 失真」
# 差一個數量級，故三份清單分開給——半徑由 `warp_radius_calibration.py` 在本機
# （不跑 GPU）先量出來，讓每一條曲線都跨過失真錨點；`matched_distortion_table.py`
# 拒絕外插，半徑猜錯就是整批白跑，而卡是多人共用的。
#
# **固定強度下比不同條件是錯的**（先前有三個結論這樣出錯過），出表一律用
# `matched_distortion_table.py --anchor`。
#
# 用法：bash scripts/warp_triad.sh "<卡號>" [all|opt|opt_paired|cheap|rand|trip] ["<半徑>"] ["<③ 的半徑>"]
#   cheap      = rand ＋ trip，兩個不最佳化的條件，剛好一張卡的兩個 slot
#   opt_paired = opt 只用一張卡：兩個 process 各跑一半的半徑，總時數加倍
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_warp
mkdir -p "$OUT"

# 主線的十三張。**檔案不在就直接停**，不要退回「跑全部」或「跑零張」——
# 遠端是 sparse-checkout 的白名單（OPERATIONS 的坑四），這個檔很容易不在，
# 而 `--images` 拿到空字串時 argparse 只會抱怨參數，訊息與「清單錯了」
# 完全不像。實測踩過一次。
IMGLIST=runs/ip2p_fair_comparison/images13.txt
if [ ! -s "$IMGLIST" ]; then
  echo "錯誤：找不到影像清單 $IMGLIST（或它是空的）。" >&2
  echo "      遠端用 sparse-checkout，這個檔可能沒被撿下來；" >&2
  echo "      從本機 sed 's/\r$//' 送過去，不要在遠端 git pull。" >&2
  exit 2
fi
IMGS=$(tr '\n' ' ' < "$IMGLIST")
N_IMGS=$(echo $IMGS | wc -w)
if [ "$N_IMGS" -ne 13 ]; then
  echo "錯誤：$IMGLIST 有 $N_IMGS 張，主線是 13 張。" >&2
  exit 2
fi
COMMON="--data data/omniedit150 --steps 1000"
# **① 的損失不是 `latent_norm`，這是被實測逼出來的改動，不是選擇。**
#
# `latent_norm`（＝ DCT-Shield §4.2 的 `‖E(x′)‖₂`，本批原訂的損失）對位移場
# **在零位移處有一個帶折點的局部極小**：`runs/ip2p_warp/step_probe_latent_norm.csv`
# 逐步量到梯度完全正常（absmean 3.4e−2、零元素比例 0.0000），但每走一步
# **損失都上升**（105.95 → 110.07），於是 sign PGD 在 0 與 ±α 之間形成週期 2
# 的振盪、`|c|` 恆等於 α。第一輪就是這樣跑出 `dists=0.0001`／`effect=0.0005`
# ——1000 步跑完防禦圖與原圖幾乎逐位相同。**照那個數字寫「最佳化買不到東西」
# 會得到一個假的否證。**
#
# 換成本專案自己的 `encoder_target`（`‖E(x′) − E(y)‖²`）之後同一支探針顯示
# 損失單調下降（0.9509 → 0.9058）、`|c|` 一路長到 1.11 px
# （`step_probe_encoder_target.csv`）。故 ① 用 `encoder_target`。
#
# **② 與 ③ 不最佳化（`params()` 回傳空 list），損失對它們沒有作用**，
# 兩邊的防禦圖與 `--loss` 無關；CSV 上的 `loss` 欄因此在三格之間不同，
# 這是標籤差異不是條件差異，報表上要註明。
LOSS_OPT="--loss encoder_target"
LOSS_RAND="--loss latent_norm"

DEVS=(${1:-0 1})
WHICH="${2:-all}"

# 預設半徑：本機 `warp_radius_calibration.py` 量出來的（不跑 GPU），三條曲線
# 都跨過工作點的 DISTS 0.1286。**不是猜的**，逐點見
# `runs/ip2p_warp/radius_calibration_rand.csv` 與 `..._roundtrip.csv`：
#
#   warp_rand       r=3 → 0.0777、r=4 → 0.1032（錨點約在 r=5）
#   warp_roundtrip  r=4 → 0.0365、r=8 → 0.1117（錨點約在 r=8.4）
#
# 兩族差一倍以上，因為往返把幾何抵消掉大半，同一個半徑上的失真低得多。
# ① 的半徑比 ② 高：sign PGD 不會把 `c` 每一格都推到邊界（實測 `|c|` 平均只到
# 半徑的三分之一左右），同一個半徑上的失真因此低於隨機場，要更大的半徑才跨得
# 過錨點。同樣的現象在候選二（`runs/ip2p_shading`）上也出現過。
OPT_RADII=(${3:-4 8 16 24})
RAND_RADII=(${3:-3 4 6 8})
# 第四個參數只給 ③。三格的「半徑 → 失真」差一倍以上，`cheap` 一次送兩格時
# 常常要給兩份不同的清單；沒有這個參數就只能分兩次呼叫，而第一次送出之後
# 那張卡就不再是空的，第二次會（正確地）被 `--assert` 擋下來。
TRIP_RADII=(${4:-${3:-4 8 12 16}})

require_slots() {
  local n_points="$1" n_devs="$2"
  if [ "$n_points" -gt $(( n_devs * 2 )) ]; then
    echo "錯誤：$n_points 個工作點需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
    echo "      只給了 $n_devs 張。每卡最多 2 個 process。" >&2
    exit 2
  fi
}

# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

# 工作點數：`opt` 每個半徑一個 process（1000 步 PGD，貴）；`rand` 與 `trip`
# 不最佳化（`params()` 回傳空 list，`run_param_pgd` 直接回傳），一個 process
# 就把整串半徑跑完，省下每次約 320 秒的模型載入。
POINTS=0
case "$WHICH" in
  all)  POINTS=$(( ${#OPT_RADII[@]} + 2 )) ;;
  opt)  POINTS=${#OPT_RADII[@]} ;;
  rand|trip) POINTS=1 ;;
  # 兩個不最佳化的條件合起來剛好是一張卡的兩個 slot。分兩次呼叫送不出去：
  # 第一次送完之後那張卡的已用記憶體就超過 `free_cards.sh` 的門檻，第二次
  # 的 `--assert` 會（正確地）拒絕。
  cheap) POINTS=2 ;;
  # 卡不夠時的 opt：兩個 process，各自把一半的半徑依序跑完。總時數加倍，
  # 但只佔一張卡的兩個 slot。卡是多人共用的，寧可慢也不要擠。
  opt_paired) POINTS=2 ;;
  *) echo "未知的第二個參數 '$WHICH'（要 all|opt|opt_paired|cheap|rand|trip）" >&2; exit 2 ;;
esac
require_slots "$POINTS" "${#DEVS[@]}"

i=0
launch_opt() {
  for rad in "${OPT_RADII[@]}"; do
    tag="opt_r$(echo "$rad" | tr -d '.')"
    dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
        --out "$OUT/$tag" --conditions warp --radius "$rad" \
        --images $IMGS $COMMON $LOSS_OPT > "$OUT/$tag.log" 2>&1 &
    echo "[warp] $tag cond=warp radius=$rad dev=$dev pid=$!"
  done
}

launch_serial() {   # $1 = 條件名, $2 = tag 前綴, $3 = runner 檔名標籤, 其餘 = 半徑
  # `rand` 與 `trip` 不最佳化（`params()` 回傳空 list），一格只花一次前向
  # 加一次編輯，故整串半徑在同一個 process 裡依序跑，省下每次約 320 秒的
  # 模型載入。**產生一支 runner 落地再 nohup 它**，不用行內子 shell：
  # 一來 ssh 斷線時 SIGHUP 不會把後面的半徑安靜地帶走，二來實際送出去的
  # 指令留在檔案裡看得到。
  local cond="$1" short="$2" label="$3"; shift 3
  local LOSS="$LOSS_RAND"
  [ "$cond" = warp ] && LOSS="$LOSS_OPT"
  local dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  # **檔名要帶上半徑**。固定檔名會被下一次呼叫覆寫，而 bash 是**邊讀邊執行**
  # 的：上一批還在跑的 runner 會從舊的位元組位移繼續讀新內容，之後執行到的
  # 是一段被切斷的指令，沒有任何症狀。實測踩過一次——第二次呼叫 `cheap` 把
  # 第一次的 `rand_runner.sh` 蓋掉，兩個還在跑的 runner 只好整個停掉重來。
  local radtag=$(echo "$*" | tr ' .' '-_')
  local runner="$OUT/${label}_r${radtag}_runner.sh"
  if [ -e "$runner" ] && pgrep -f "$runner" > /dev/null; then
    echo "錯誤：$runner 正在被執行，拒絕覆寫。" >&2
    exit 2
  fi
  {
    echo "#!/usr/bin/env bash"
    echo "cd \"$ROOT\""
    for rad in "$@"; do
      local tag="${short}_r$(echo "$rad" | tr -d '.')"
      # 一行一格，不折行——折行要在**產生的**檔裡放反斜線，很容易被外層
      # 的 shell 吃掉，而吃掉之後三個 echo 會併成一個。
      echo "CUDA_VISIBLE_DEVICES=$dev \"$PY\" scripts/ip2p_run.py --out \"$OUT/$tag\" --conditions $cond --radius $rad --images $IMGS $COMMON $LOSS > \"$OUT/$tag.log\" 2>&1"
    done
  } > "$runner"
  nohup bash "$runner" > "$OUT/${label}_r${radtag}_driver.log" 2>&1 &
  echo "[warp] $label（$cond）半徑 $* 依序跑 dev=$dev pid=$!"
}

case "$WHICH" in
  all)  launch_opt
        launch_serial warp_rand rand rand "${RAND_RADII[@]}"
        launch_serial warp_roundtrip trip trip "${TRIP_RADII[@]}" ;;
  opt)  launch_opt ;;
  opt_paired)
        launch_serial warp opt opt_low "${OPT_RADII[@]:0:2}"
        launch_serial warp opt opt_high "${OPT_RADII[@]:2}" ;;
  cheap) launch_serial warp_rand rand rand "${RAND_RADII[@]}"
         launch_serial warp_roundtrip trip trip "${TRIP_RADII[@]}" ;;
  rand) launch_serial warp_rand rand rand "${RAND_RADII[@]}" ;;
  trip) launch_serial warp_roundtrip trip trip "${TRIP_RADII[@]}" ;;
esac
echo "[warp] 全部送出（$(date)），共 $i 個 process"
