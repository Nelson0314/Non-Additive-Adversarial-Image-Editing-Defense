#!/usr/bin/env bash
# 內容約束的頭對頭：幾何固定，只有「補丁裡長什麼樣子」在變。
#
# 這一批問三件事
#   一、梯度到底買到什麼。`rand` 與 `free` 的幾何逐位元相同，差別只有那六千步。
#       隨機若打平，23% 的成功就只是「衣服被蓋住了」而不是防禦學到東西。
#   二、低頻要怎麼綁才不像雜訊。`tint` 是懲罰（要調 λ、與主損失搶預算），
#       `lowproj` 是約束（低頻在夾取前恆等於原圖），`chroma` 更強
#       （亮度逐位元不動，只有顏色可學）。
#   三、色彩重映射在新讀數下如何。它此前只用位移量過（判為 12–14%，不成立），
#       而位移量不到「還認不認得出是誰」——那正是這一批換掉的讀數。
#
# 為什麼防禦只跑一次而攻擊跑三次
#   `L_ig` 只吃**空字串**的文字嵌入，攻擊指令要到編輯那一步才進來，故同一組
#   旗標在三類指令下解的是**完全相同**的最佳化問題。先前那批把它跑了三遍，
#   三張防禦圖兩兩差 24–26 灰階、目標值最多差 137%——那是最佳化本身的離散，
#   不是條件的差別。這裡改成訓練一次（clothing）、另外兩類用 `--steps 0`
#   重播同一張防禦圖，機時省三分之二，而且三類之間不再混著那一層離散。
#
# 用法：bash scripts/content_constraint_round.sh "<卡號…>" <臂> [影像前綴]
#       臂：內容軸 rand／tint／lowproj／lowproj_tile／chroma／colorgrid／colorgrid_rand／lattice／lattice_rand／lat6／ring／ring_rand／identity／identity_layout／eot／attn
#           位置軸 spread／blocks（＋各自的 _rand），面積逐張對齊衣物臂
#                  bg／bg_rand 是舊的侵蝕背景，形狀被人形輪廓帶著走，已不用
#       （`free` 不在這裡：`runs/ip2p_face_defence/clothing_plain_*` 就是它，
#         同一組旗標、同一個種子，重跑只會多花機時。）
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
ARM="${2:-}"
ONLY="${3:-}"
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" <臂> [影像前綴]" >&2; exit 2; }

# **一次最多五張卡**（CLAUDE.md）。多給的裁掉而不是照單全收——這是多人共用
# 的機器，把八張卡佔滿會讓別人完全排不進來。
if [ ${#DEVS[@]} -gt 5 ]; then
  echo "提醒：給了 ${#DEVS[@]} 張卡，裁到前五張（${DEVS[*]:0:5}）"
  DEVS=("${DEVS[@]:0:5}")
fi

# 三類指令共用同一張防禦圖：CAT_TRAIN 那一類負責訓練並存權重，其餘重播。
CAT_TRAIN=clothing
CATS_REPLAY="accessory background"

case "$ARM" in
  # 同幾何、隨機內容、零最佳化。`patch_rand` 的 `params()` 是空的，
  # `run_param_pgd` 走 no_params 分支直接回傳，一格約一分鐘。
  rand)        COND=patch_rand; EXTRA="" ;;
  # 低頻**懲罰**。現行旋鈕，放在 23% 的面積上（先前只在 5% 那組開過）。
  tint)        COND=patch;      EXTRA="--patch-tint 3.0" ;;
  # 低頻**替換**。σ 與 tint 同一個尺度、共用同一個盒式模糊。
  lowproj)     COND=patch;      EXTRA="--patch-lowfreq 16" ;;
  # 低頻替換 ＋ 高頻平舖。**兩個正交的帶**：低頻跟著衣服走（顏色與明暗），
  # 高頻由一塊 64×64 的磚重複鋪滿（規則重複是浮水印的第一視覺特徵）。
  # 單獨的 lowproj 拉圖看到「衣服上灑了彩色紙屑」——低頻對了但高頻仍是逐像素
  # 雜訊，因為那一帶完全自由。這一格問的是把它也綁住之後還剩多少效果。
  lowproj_tile) COND=patch;     EXTRA="--patch-lowfreq 16 --patch-tile 64" ;;
  # 凍結亮度，只有色度可學。褶皺與陰影是原本那件衣服的。
  chroma)      COND=patch;      EXTRA="--patch-chroma" ;;
  # 色彩重映射鎖在衣服上。半徑與格數取 `runs/ip2p_color_hunt/` 的最佳工作點。
  colorgrid)   COND=color_grid; EXTRA="--color-grid 16 --color-luma-bins 16" ;;
  colorgrid_rand)
               COND=color_grid_rand; EXTRA="--color-grid 16 --color-luma-bins 16" ;;
  # 位置那一軸：同樣大小的花紋改放到**臉以外的地方**。面積逐張對齊該張影像
  # 衣物支撐的面積（下面的 TARGET），否則等於同時動了「放在哪裡」與「放多大」
  # 兩個變因。
  #
  # `bg` 用 ATR 的背景類別再侵蝕到同面積，長出來的是一個貼著人形輪廓的怪形狀
  # ——那時變的不只是位置，還有形狀。下面兩個改用**與語意無關的支撐**：
  #   spread  整張畫面（臉以外）均勻鋪滿，權重調淡到同面積：一大片各改一點
  #   blocks  六個等大的圓斑分散在臉以外的地方：固定形狀、幾塊、位置分散
  # 兩者的 mean(w) 與衣物臂完全相同，差別只有「攤開還是集中」與「放在哪裡」。
  bg)          COND=patch;      EXTRA="" ;;
  bg_rand)     COND=patch_rand; EXTRA="" ;;
  spread)      COND=patch;      EXTRA="--carrier-match scale" ;;
  # 浮水印點陣：間距 8 像素（＝一個 latent 格）、半徑 2 的小圓斑。
  # token 覆蓋率 100% 而像素面積只有 18.75%——比衣物臂的 23% 還小。
  # 檢驗的假說是「效果由 token 覆蓋率決定，不是像素面積」。
  # `lat6` 是同一個假說的極端點：半徑 1.5、面積只有 6.25%，覆蓋率仍然是滿的。
  # 若它也成立，那 `pretty`（5% 面積、0/10）失敗的原因就確定是覆蓋率而非面積。
  lattice)     COND=patch;      EXTRA="--carrier-lattice 8 --carrier-dot-radius 2.0" ;;
  lattice_rand) COND=patch_rand; EXTRA="--carrier-lattice 8 --carrier-dot-radius 2.0" ;;
  lat6)        COND=patch;      EXTRA="--carrier-lattice 8 --carrier-dot-radius 1.5" ;;
  # ── 五個新機制 ───────────────────────────────────────────────────
  # ring：緊貼臉的環帶。賭「碰到**對的** token」而不是「碰到所有 token」。
  ring)        COND=patch;      EXTRA="--carrier-ring 24 --carrier-ring-inner 8" ;;
  ring_rand)   COND=patch_rand; EXTRA="--carrier-ring 24 --carrier-ring-inner 8" ;;
  # identity：把判準寫進損失。此前訓練最小化代理量而判準是身分，兩者已分歧。
  identity)    COND=patch;      EXTRA="" ;;
  # identity_layout：雙目標。要求「同一個場景、換一張臉」而不是整張圖被毀。
  identity_layout) COND=patch;  EXTRA="" ;;
  # eot：對攻擊指令的分布取期望。現行損失只見過空字串。
  eot)         COND=patch;      EXTRA="--prompt-eot" ;;
  # attn：把編輯指令的注意力吸到補丁上（改道而非破壞）。
  attn)        COND=patch;      EXTRA="--attn-weight 1.0" ;;
  spread_rand) COND=patch_rand; EXTRA="--carrier-match scale" ;;
  blocks)      COND=patch;      EXTRA="--carrier-scatter 6" ;;
  blocks_rand) COND=patch_rand; EXTRA="--carrier-scatter 6" ;;
  *) echo "第二個參數必須是 rand／tint／lowproj／lowproj_tile／chroma／colorgrid／colorgrid_rand／lattice／lattice_rand／lat6／ring／ring_rand／identity／identity_layout／eot／attn／bg／bg_rand／spread／spread_rand／blocks／blocks_rand" >&2
     exit 2 ;;
esac
case "$ARM" in
  bg|bg_rand) CARRIER="background"; EXTRA="$EXTRA --carrier-max-area 0.95" ;;
  spread|spread_rand|blocks|blocks_rand|lattice|lattice_rand|lat6|ring|ring_rand)
              CARRIER="frame";      EXTRA="$EXTRA --carrier-max-area 1.0" ;;
  *)          CARRIER="clothes" ;;
esac
# 損失軸：這幾個臂換的是**損失**不是支撐，故載體維持衣物、只改 --loss。
case "$ARM" in
  identity)        LOSS="identity" ;;
  identity_layout) LOSS="identity"; EXTRA="$EXTRA --id-layout-weight 1.0" ;;
  *)               LOSS="image_guidance" ;;
esac

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_content_constraint
mkdir -p "$OUT"

# 影像清單由 **`free` 臂實際跑過的那些格**推導，不是由目錄檔。
#
# 目錄檔 `data/attack_prompts.yaml` 登記了 20 張，而 `free`（＝
# `runs/ip2p_face_defence/clothing_plain_*`）只跑了其中 10 張。這一批的每一個臂
# 都要與 `free` 逐張並排，用目錄檔就會多跑 10 張沒有對照的格——那不會報錯，
# 只會讓一半的列在比較表上是空的。由對照組的目錄反推，「同一批影像」就成為
# 構造保證而不是靠人記得。
#
# 同時檢查每一張都在目錄檔裡有指令，缺了就整批拒絕（`--attack-category` 那時
# 會逐張死掉，但那是跑到一半才死）。
IMGS=$("$PY" - <<'PYEOF'
import pathlib, sys
import yaml

spec = yaml.safe_load(open("data/attack_prompts.yaml", encoding="utf-8"))
have = set(spec["images"])
root = pathlib.Path("runs/ip2p_face_defence")
names = sorted(p.name[len("clothing_plain_"):]
               for p in root.glob("clothing_plain_task_*") if p.is_dir())
if not names:
    sys.exit(f"{root} 底下沒有 clothing_plain_* 目錄：對照組不在就沒有可比的基準")
missing = [n for n in names if n not in have]
if missing:
    sys.exit(f"這些影像有對照組卻沒有攻擊指令：{missing}")
print(" ".join(names))
PYEOF
) || { echo "錯誤：影像清單推導失敗（見上）" >&2; exit 2; }
[ -z "$IMGS" ] && { echo "錯誤：推導不出影像清單。" >&2; exit 2; }
echo "影像 $(echo $IMGS | wc -w) 張（與 free 臂同一批）"

SEL=""
for img in $IMGS; do
  if [ -n "$ONLY" ]; then case "$img" in $ONLY*) ;; *) continue ;; esac; fi
  SEL="$SEL $img"
done
N=$(echo $SEL | wc -w)
[ "$N" -eq 0 ] && { echo "錯誤：過濾器 '$ONLY' 一張都沒對上。" >&2; exit 2; }
# 每卡最多兩個 process。
if [ "$N" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$N 張影像需要至少 $(( (N + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

# 色彩族的載體是 `apply_where`，補丁族的載體是**支撐**；兩者都由 ATR 的衣物
# 類別給，而受保護主體都走 `--subject-source face`，故兩族的「衣服」與「臉」
# 是同一個物件，不是兩份各自演化的實作。
COMMON="--data data/omniedit150 --conditions $COND \
--attack-prompts data/attack_prompts.yaml \
--subject-source face --loss $LOSS --ig-zt diffuse_src \
--patch-carrier $CARRIER --carrier-refine 4 --carrier-erode 3 --carrier-feather 8 \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002"
case "$COND" in
  patch|patch_rand) COMMON="$COMMON --patch-placement complement --radius 0.05" ;;
  # 色彩族的 radius 是網格仿射係數對單位矩陣的 L∞ 偏移，與補丁的面積比例
  # **不可互相換算**；0.30 是 `runs/ip2p_color_hunt/` 的工作點。
  *)                COMMON="$COMMON --radius 0.30" ;;
esac

i=0
for img in $SEL; do
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  TRAIN_OUT="$OUT/${ARM}_${CAT_TRAIN}_$img"
  # 背景那兩臂逐張對齊面積。目標值取**那張影像衣物支撐的實際面積**，
  # 由已跑完那批的 CSV 讀出來——寫死一個數字的話，十張圖會共用同一個面積，
  # 而「同面積」正是這個對照要控制的變因。讀不到就整格拒絕，不要猜一個。
  AREA_FLAG=""
  # 點陣的面積是 pitch 與 radius 的函數，**不對齊**（對齊會破壞「每個 latent
  # 格恰好被碰到一次」的構造，驅動的守門也會擋）。故它不進這個 case。
  case "$ARM" in bg|bg_rand|spread|spread_rand|blocks|blocks_rand)
    TARGET=$("$PY" - "$img" <<'PYEOF'
import csv, pathlib, sys
p = pathlib.Path("runs/ip2p_face_defence") / f"clothing_plain_{sys.argv[1]}" / "results.csv"
if not p.exists():
    sys.exit(f"讀不到 {p}：背景臂的面積要對齊衣物臂，來源不在就不能猜")
rows = list(csv.DictReader(p.open(encoding="utf-8")))
if len(rows) != 1:
    sys.exit(f"{p} 有 {len(rows)} 列，預期恰好 1 列")
print(rows[0]["patch_area"])
PYEOF
) || { echo "錯誤：$img 取不到對齊面積" >&2; exit 2; }
    # `blocks` 的面積由 `scatter_support` 吃 `--radius` 達成（它二分搜尋半徑
    # 直到實現面積等於要求），故那兩臂覆寫 radius 而不給 target-area；其餘走
    # `--carrier-target-area`。**兩條路不可以同時給**：那會先散成圓斑再整片
    # 調淡，面積被縮兩次而 CSV 上看不出來。
    case "$ARM" in
      blocks|blocks_rand) AREA_FLAG="--radius $TARGET" ;;
      *)                  AREA_FLAG="--carrier-target-area $TARGET" ;;
    esac
    echo "[content] $ARM $img 面積對齊到 $TARGET"
  ;; esac
  # 一張卡上的一條鏈：先訓練（存權重），再用同一張防禦圖重播另外兩類指令。
  # **序列而不是並行**：重播吃的是訓練那一格存下的 `__w.pt`，並行會讀到
  # 還沒寫出來的檔，而 `--steps 0` 的守門那時才會擋下來、整格白排。
  # 不最佳化的臂**不走重播**：它們的 `params()` 是空的，存出來的權重檔也是
  # 空的，而重播的價值本來就在省下那六千步——沒有那六千步就沒有東西可省，
  # 反而多一條會靜默載到空清單的路徑。那三類各自跑滿，一格約一分鐘。
  case "$COND" in
    *_rand) SAVE=""; REPLAY_STEPS=6000; RESUME="" ;;
    *)      SAVE="--save-weights"; REPLAY_STEPS=0
            RESUME="--resume-weights $TRAIN_OUT" ;;
  esac
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
    set -e
    '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN --steps 6000 --step-size 0.01 \
        $SAVE $COMMON $EXTRA $AREA_FLAG
    for cat in $CATS_REPLAY; do
      '$PY' scripts/ip2p_run.py --out '$OUT/${ARM}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat --steps $REPLAY_STEPS --step-size 0.01 \
          $RESUME $COMMON $EXTRA $AREA_FLAG
    done
  " < /dev/null > "$OUT/${ARM}_$img.log" 2>&1 &
  disown
  echo "[content] $ARM $img dev=$dev（訓練 $CAT_TRAIN，重播 $CATS_REPLAY）"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"
