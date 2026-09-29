#!/usr/bin/env bash
# 臂名 → 產防禦圖的指令。現行只有 color（參數定義在 lab/code/color_defence.py）。
#
# 用法：bash lab/scripts/defence_cmd.sh color [額外參數…]
#        產出寫到 lab/runs/defence/color/；DEF_OUT 可覆寫（分片執行用，
#        results.csv 每寫一列就整份重寫，不可共用目錄）。額外參數附在指令最後。
set -uo pipefail
ARM="$1"
OUT="${DEF_OUT:-lab/runs/defence/$ARM}"
C=("$PY" lab/code/color_defence.py --arm "$ARM" --out "$OUT" --data lab/data/portraits)
case "$ARM" in
  color)                exec "${C[@]}" "${@:2}" ;;
  # 實驗臂：simple＝色偏上限改為錨點方框、只留膚色同色上限；xattn＝降低對類別詞的注意力；
  # skinbox＝冷色方向的方框依離膚色中心的距離放大到 2 倍
  color_simple)         exec "${C[@]}" --caps simple "${@:2}" ;;
  color_simple_xattn)   exec "${C[@]}" --caps simple --objective xattn "${@:2}" ;;
  color_simple_skinbox) exec "${C[@]}" --caps simple --box skin "${@:2}" ;;
  color_xattn)          exec "${C[@]}" --objective xattn "${@:2}" ;;
  # DAYN 式主體區域注意力抑制：600 步、lr 0.01 → 0.001、乘子每 10 步更新
  color_simple_dayn)    exec "${C[@]}" --caps simple --objective dayn --steps 600 --lr 0.01 \
                          --lr-final-ratio 0.1 --lam-every 10 "${@:2}" ;;
  # 同上，載體換成耦合的 3D Lab 映射（Lut3D）
  color_lut3d_dayn)     exec "${C[@]}" --caps simple --carrier lut3d --objective dayn --steps 600 --lr 0.01 \
                          --lr-final-ratio 0.1 --lam-every 10 "${@:2}" ;;
  *)
    echo "未知的臂：$ARM" >&2; exit 2 ;;
esac
