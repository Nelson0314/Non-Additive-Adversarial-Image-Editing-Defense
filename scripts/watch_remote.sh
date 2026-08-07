#!/usr/bin/env bash
# 遠端工作的看門程序。**在本機執行**，不是在伺服器上。
#
#   bash scripts/watch_remote.sh <session1> [session2 ...]
#
# 例：
#   bash scripts/watch_remote.sh wacv-calib v14calib
#   bash scripts/watch_remote.sh wacv-bird_03 wacv-cat_02 wacv-dog_03
#
# ---------------------------------------------------------------------------
# 為什麼需要它
#
# 訓練一格要 2–3 小時，而「有沒有在跑」與「有沒有在算」是兩件事。2026-08-06
# 實測遇過三種只看 `tmux ls` 看不出來的狀態：
#
#   1. session 還在、行程已死（被 kill 之後留下 `.writer.lock`，下一次啟動
#      會以 `WriterLockHeld` 拒絕，而外面只看到「沒起來」）
#   2. 行程還在、卡佔著、但 log 不再推進
#   3. 訓練跑完才在存檔或指標那一行以例外中止（當天發生四次）
#
# 故本腳本每 90 秒探一次，並在下列任一情形**立刻**回報並結束：
#
#   收工（log 出現 `[exit`）／例外（出現 `Traceback`）／session 消失
#
# 都沒事則在第 6 次（約 9 分鐘）回報一次心跳，並附上 log 是否推進。
# 迴圈長度受呼叫端的 timeout 上限約束（10 分鐘），故 6 × 90 秒是可取的最長值；
# 要持續監看就在收到回報後重新執行一次。
#
# **回報只是通知，不會停掉遠端的工作。** 停掉要自己 `tmux kill-session`。
# ---------------------------------------------------------------------------
set -uo pipefail

H="${WACV_HOST:-nelson0314@server.basiclab.lab.nycu.edu.tw}"
PORT="${WACV_PORT:-10101}"
JOBS="${*:?用法：bash scripts/watch_remote.sh <session> [session ...]}"

probe() {
ssh -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$H" \
  "JOBS='$JOBS' bash -s" <<'EOS'
R=$HOME/wacv_runs
for S in $JOBS; do
  # session 名稱與 log 路徑的對應。`shard.sh` 的分片是 wacv-<批次>-<影像>，
  # 對到 $R/<批次>_<影像>/run.log；段 0 是 wacv-calib（b3）與 v14calib。
  # 批次名不含 `-`、影像 id 不含 `-`，故第一個 `-` 就是兩者的分界。
  case $S in
    wacv-calib)  L=$R/b3/calib.log;;
    v14calib)    L=$R/v14/calib.log;;
    wacv-*-*)    T=${S#wacv-}; L=$R/${T%%-*}_${T#*-}/run.log;;
    wacv-*)      L=$R/b3_${S#wacv-}/run.log;;   # 舊式命名，b1／b2 的既有 log
    b2-*)        L=$R/b2_${S#b2-}/run.log;;
    # 診斷用的一次性 session：`sweep_<批次>` 對到該批次的 sweep.log。
    # 2026-08-07 加入。before：落到下面的 `*)` 而找到 $R/sweep_v14/run.log
    # 這個不存在的路徑，`wc -l` 於是把 stderr 印進本函式的輸出，整輪回報作廢。
    sweep_*)     L=$R/${S#sweep_}_bird_03/sweep.log;;
    *)           L=$R/$S/run.log;;
  esac
  alive=no; tmux has-session -t "$S" 2>/dev/null && alive=yes
  done_=no; tail -5 "$L" 2>/dev/null | grep -q '\[exit' && done_=yes
  err=no;  grep -ql 'Traceback' "$L" 2>/dev/null && err=yes
  # 推進與否看**逐格紀錄**而不是 log 行數。
  #
  # 2026-08-07 改。before：`n=$(wc -l < "$L")`。`run_stage` 每 50 格才印一行，
  # 而 b3（SDXL/1024²）一格要 20 秒以上——50 格即 17 分鐘，遠超本腳本 9 分鐘
  # 的窗口，於是正常跑的分片被連續回報為「可能卡住」。逐格紀錄每格都寫，
  # 是這裡唯一夠細的訊號。
  D=$(dirname "$L")
  n=$(ls "$D"/_cells/*.json 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && n=$(wc -l < "$L" 2>/dev/null || echo 0)   # 段 0 沒有 _cells
  echo "$S|$alive|$done_|$err|$n|$(tail -1 "$L" 2>/dev/null | cut -c1-60)"
done
EOS
}

prev=""; stall=0; cur=""
for i in $(seq 1 6); do
  cur=$(probe)
  if [ -z "$cur" ]; then echo "SSH 探測失敗（第 $i 次）"; sleep 90; continue; fi
  if echo "$cur" | awk -F'|' '$2=="no"||$3=="yes"||$4=="yes"{f=1} END{exit !f}'; then
    echo "=== 狀態改變（收工／例外／session 消失）==="; echo "$cur"; exit 0
  fi
  keys=$(echo "$cur" | cut -d'|' -f1,5)
  if [ "$keys" = "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  prev="$keys"
  sleep 90
done

if [ "$stall" -ge 5 ]; then
  echo "=== 心跳：log 連續 9 分鐘沒有推進。可能只是步慢，也可能卡住 ==="
else
  echo "=== 心跳：正常推進 ==="
fi
echo "$cur"
# 欄位：session|存活|已收工|有例外|log 行數|最後一行
