# 交接

程式與數值在 `non-additive-frequency/`。工作規則見該目錄的 `CLAUDE.md`。

## 現況

遠端沒有本專案的行程，本機沒有背景工作。

測試基準：`python -m pytest -q --ignore=runs` → **1331 passed / 2 skipped /
1 xfailed**。`--ignore=runs` 是必要的，`runs/ncf_cpu_test_tmp/pytest-of-nelso`
被提權沙箱的 ACL 鎖住，會讓收集階段直接失敗。

本機 Python `C:/Users/nelso/miniconda3/envs/wacv/python.exe`。
遠端工作目錄 `/nfs/home/nelson0314/WACV-s3`（basic-1 埠 10101）。

## 跑過什麼

`runs/objective_pilot/`：五張人像 × 三類指令 = 15 格，四個臂
（`latent_norm`、`latent_norm` 邊界起點、`cfg_shift`、同可達集合隨機對照）。
設定、目錄、量測欄位與中位數在該目錄的 `README.md`。

`runs/subject_anchored_readout/`：主體錨定身分讀數與盲評的聯結，
含各讀數對人眼判定的 AUC。

`runs/lowfreq/`：1120 列，但全部是 `max_gain=1.0`、`T_distance=0`、未訓練，
等於恆等映射，且只有代理讀數（`clip_margin`）。幅度那一軸沒有取樣過。

## 下一步：顏色族的天花板，卡在遠端怎麼放程式

計畫 `docs/superpowers/plans/isometric-color-field.md`。程式與測試都寫完了
（本機 1367 passed / 2 skipped / 1 xfailed），**批次還沒派**。

派工 `scripts/color_ceiling.py --config configs/color_ceiling.json`，
臂 × ΔE00 掃描，不訓練，約 2.6 GPU-hours（15 格 × 每格 30 次編輯 × 21 秒）。
設定與量測欄位見 `non-additive-frequency/runs/color_ceiling/README.md`。

**卡住的地方：遠端那棵樹與本機分岔了。** `/nfs/home/nelson0314/WACV-s3` 在
commit `e907913fe`，目錄結構是舊的（`docs/` 在頂層，不在
`non-additive-frequency/` 之下），約 150 支 `scripts/` 有未提交的修改，而
`src/defense/naturalness_gate.py` 的內容與本機**不同**（遠端
`8ac8306f…`、本機 `9cf17277…`）。把新檔案 rsync 上去會讓新程式跑在舊版的
`gate_row`／`subject_identity_row` 上，讀數可能靜默不可並列。三個選項見下方
「要決定的事」。

卡的狀態（basic-1）：1、5、6 三張沒有別人的 compute app 且記憶體 4 MiB。

## 要決定的事

遠端怎麼放這批程式：

1. **開新目錄**（建議）：從本機同步一份到 `/nfs/home/nelson0314/WACV-s4`，
   完全不碰現有那棵樹。代價是多一份磁碟與一次環境確認。
2. **只複製新檔案**到 WACV-s3：最省事，但新程式會跑在遠端舊版的支援模組上。
3. **整棵同步**到 WACV-s3：覆蓋掉遠端那 150 支未提交的修改，難回復。

## 兩個會靜默失效的環境事實

- `IP2PWrapper(dtype=fp16)` 下 `resolve_precision` 讓 backbone 半精度而 VAE
  留在 fp32，官方 pipeline 的 `edit()` 會在 `vae.encode` 拋 Half/float 不符；
  `edit_differentiable` 自行處理故不受影響。要跑攻擊評測就用 fp32。
- `pkill -f <樣式>` 會匹配到 ssh 遠端命令自己的 shell，把連線殺掉並回傳 127。
  `[o]` 括號寫法只擋得住 grep 本身。改用
  `ps -u $USER -o pid=,comm=,args= | awk '$2=="python" && /<樣式>/ {print $1}'`
  取 pid 再殺，殺完回頭用 `nvidia-smi` 複驗。

## 三件寫成測試釘住的性質

1. `ReColorAdvParam.project()` **不是幂等的**：`clamp(0,1)` → CIELUV↔RGB 色域
   往返 → L∞ 盒，最後一步會把點推回色域外。與原文 `project_params` 同序，不要修。
2. **邊界初始化到不了盒角**：色域投影把格點邊緣拉回來，可達集合不是盒子。
3. `run_param_pgd` 在讀 `params()` **之前**會先 `param.reset(x01, seed)`。
   訓練前做的初始化會被抹掉，且不拋錯。

## 使用者的裁定

1. 可見的防禦是允許的，只要不動主體、產物自然。
2. 只接受兩種載體：整圖顏色濾鏡、衣物換色。
3. 分工：衣物指令由衣物載體處理，頭部與背景指令由整圖濾鏡處理。
4. 載體不得往高頻加東西（優先於其餘各項）。
5. 畫面崩壞也算防禦成功——判準是攻擊者有沒有拿到一張可用的、認得出是同一個人
   的、已完成指令的照片。
6. 助手不做成敗判斷，只把數據與圖擺出來。
