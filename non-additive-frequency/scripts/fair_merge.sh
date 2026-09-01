#!/usr/bin/env bash
# 把分片批次併成一個目錄，供 `phase_retention.py --run` 使用（它要的是單一
# 目錄，裡面同時有 results.csv 與全部的 *__def.png）。
#
# **輸出目錄不可被自己的 glob 匹配到**（FND-062 記過這個坑：合併結果被下一輪
# 的 glob 吃回去，數字靜默翻倍）。故來源寫成 `<src>/*/`、輸出放在 `<src>` 的
# 同層而不是底下，並在開頭明確擋掉。
#
# 用法：bash scripts/fair_merge.sh runs/fair0820/b runs/fair0820/b_merged
set -eu
SRC=$1
DST=$2
case "$DST" in
  "$SRC"/*) echo "輸出 $DST 在來源 $SRC 底下，會被自己的 glob 吃到，拒絕"; exit 1;;
esac
rm -rf "$DST"
mkdir -p "$DST"

first=1
n=0
for d in "$SRC"/*/; do
  [ -f "$d/results.csv" ] || continue
  if [ $first -eq 1 ]; then
    head -1 "$d/results.csv" > "$DST/results.csv"
    first=0
  fi
  tail -n +2 "$d/results.csv" >> "$DST/results.csv"
  # 硬連結：NFS 上省時間也省空間，內容與來源逐位相同
  ln -f "$d"/*.png "$DST"/ 2>/dev/null || cp -f "$d"/*.png "$DST"/
  n=$((n+1))
done
rows=$(tail -n +2 "$DST/results.csv" | wc -l)
pngs=$(ls "$DST"/*.png 2>/dev/null | wc -l)
echo "併入 $n 個分片：$rows 列、$pngs 張圖 → $DST"
[ "$rows" -gt 0 ] || { echo "零列，中止"; exit 1; }
