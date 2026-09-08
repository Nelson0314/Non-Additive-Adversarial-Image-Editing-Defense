#!/usr/bin/env bash
# 三批「歸因」工作，一格一卡，全部**不訓練**。
#
# 一、`ig_probe`：各族已存的防禦圖實際把**影像引導項**壓到哪裡。
#     `runs/latent_norm_probe/` 已證實位移不是 `‖E(x_def)‖₂` 的函數——隨機
#     色彩網格的範數比原圖還高（1.067×）而位移更大。換一個對應機制的量
#     （`image_guidance` 的固定評估）重新排一次，看位移排不排得動。
#
# 二、`regional`：把位移拆成主體內／主體外。現行的全圖位移對色彩族會被
#     灌水——改掉的顏色原封不動穿過編輯。`runs/ip2p_color_masked/` 的 1.192
#     倍是不是這樣來的，分區之後直接看得出來。
#
# 三、`patch`：補丁的零號實驗。在主體之外放一塊固定內容（噪聲／中灰／
#     另一張照片），問 IP2P 有沒有**長程通道**。不最佳化、不宣稱效果。
#     依 `scripts/patch_probe.py` 的 docstring，主體內位移貼著零就結案。
#
# 三批彼此獨立，可同時送。前兩批各數分鐘，補丁那批每張影像約 38 次編輯、
# 獨佔一卡約 11 分鐘。
#
# 用法：bash scripts/loss_attribution_probe.sh "<卡號…>" [只送這些 tag 的前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
# 檢查擋在派工前面，不是印出來就算。指定的卡上有別人就拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

IMAGES="task_attr_mod_color_11699 task_attr_mod_color_6205"

# 兩支探針共用同一份來源清單。每個 --entry 是一個已存批次的輸出目錄；
# 條件名由該目錄的 results.csv 讀出，不由檔名反推。
ENTRIES="\
--entry phase_gain_r09=runs/ip2p_mainline/ours_pg_n \
--entry phase_r09=runs/ip2p_mainline/ours_ph_n \
--entry dct_shield=runs/ip2p_mainline/dct_native \
--entry phase_ln_eot=runs/ip2p_ig_loss/ln_eot \
--entry phase_ig_eot=runs/ip2p_ig_loss/ig_eot \
--entry phase_ig_noeot=runs/ip2p_ig_loss/ig_noeot \
--entry shading_r020=runs/ip2p_shading/opt_r020 \
--entry color_curve_r3=runs/ip2p_color/curve_r3 \
--entry color_ln_r010_1000=runs/ip2p_color/grid_r010 \
--entry color_rand_r0083=runs/ip2p_color/grid_rand_r0083 \
--entry color_rand_r010=runs/ip2p_color/grid_rand_r010 \
--entry color_rand_r020=runs/ip2p_color/grid_rand_r020 \
--entry color_ln_conv_plant=runs/ip2p_color_converge/latent_norm_plant \
--entry color_ln_conv_mario=runs/ip2p_color_converge/latent_norm_mario \
--entry color_ig_conv_plant=runs/ip2p_color_converge/image_guidance_plant \
--entry color_ig_conv_mario=runs/ip2p_color_converge/image_guidance_mario \
--entry color_ig_r005_plant=runs/ip2p_color_ig/ig_r005_plant \
--entry color_ig_r005_mario=runs/ip2p_color_ig/ig_r005_mario \
--entry color_ig_r020_plant=runs/ip2p_color_ig/ig_r020_plant \
--entry color_ig_r020_mario=runs/ip2p_color_ig/ig_r020_mario \
--entry color_masked_r010_plant=runs/ip2p_color_masked/m_r010_plant \
--entry color_masked_r010_mario=runs/ip2p_color_masked/m_r010_mario \
--entry color_masked_r020_plant=runs/ip2p_color_masked/m_r020_plant \
--entry color_masked_r020_mario=runs/ip2p_color_masked/m_r020_mario"

mkdir -p runs/ig_probe runs/regional_displacement runs/patch_probe

# 四個 tag 各自寫**不同的**輸出目錄——每寫一列是整份重寫 CSV，兩個 process
# 寫同一個目錄會互相蓋掉而且不報錯。
#
# 指令直接在本 shell 裡展開後背景執行，**不包成函式再丟給 `bash -c`**：
# `declare -f` 送過去的函式體不帶 `$PY`／`$ENTRIES` 這些非匯出變數，
# 展開成空字串之後 argparse 會拿到一份殘缺的指令列。
TAGS="ig_probe regional patch_plant patch_mario"
ONLY="${2:-}"

i=0
for tag in $TAGS; do
  if [ -n "$ONLY" ]; then
    case "$tag" in $ONLY*) ;; *) continue ;; esac
  fi
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  case "$tag" in
    ig_probe)
      CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ig_probe.py \
          --ig-zt diffuse_src --images $IMAGES \
          --out runs/ig_probe/results.csv $ENTRIES \
          < /dev/null > "runs/${tag}_driver.log" 2>&1 & ;;
    regional)
      CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/regional_displacement.py \
          --images $IMAGES \
          --out runs/regional_displacement/results.csv $ENTRIES \
          < /dev/null > "runs/${tag}_driver.log" 2>&1 & ;;
    patch_plant)
      CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/patch_probe.py \
          --ig-zt diffuse_src --images task_attr_mod_color_11699 \
          --out runs/patch_probe/plant \
          < /dev/null > "runs/${tag}_driver.log" 2>&1 & ;;
    patch_mario)
      CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/patch_probe.py \
          --ig-zt diffuse_src --images task_attr_mod_color_6205 \
          --out runs/patch_probe/mario \
          < /dev/null > "runs/${tag}_driver.log" 2>&1 & ;;
    *) echo "未知的 tag $tag" >&2; exit 2 ;;
  esac
  disown
  echo "[probe] $tag dev=$dev → runs/${tag}_driver.log"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]g_probe.py\|[r]egional_displacement.py\|[p]atch_probe.py') 行探針（$(date)）"
