# 交接 prompt（2026-08-02）

以下內容可直接貼進新 session。

---

WACV 專案接手 — 白盒非加性抗文字編輯防禦

## 先做這件事

讀 `docs/NEXT_SESSION.md`（2026-08-01 改寫），再讀 `CLAUDE.md` 與
`docs/INDEX.md`。要細節再讀 `docs/RESULTS_E25-E26.md`（攻擊端的根因）、
`docs/RESULTS_E27_calibration.md`（四個假的綁定者）、
`docs/RESULTS_E28_chroma.md`（色度約束）。讀完就有完整脈絡，不需要翻 git log。

## 這一階段要做的事

**在遠端 GPU 上跑校準與主網格。** 協議已全部定案，設定寫在
`docs/NEXT_SESSION.md` §4，指令寫在 §5（校準）與 §6（主網格）。

順序是死的，不要跳過第一步：

1. **先跑校準**（8 格、60 步、約 20 分鐘）。`e27d` 那輪是在還沒有色度約束時
   做的，加入第三道之後 site C 的解一定會被壓下來，lr=0.3 還適不適用是未知的。
2. **用 `scripts/e27_binding_check.py` 確認每一格的綁定者都是 LPIPS hinge**，
   不是硬上界、不是 margin、不是色度 hinge。這一步不通過就不要開主網格。
3. **主網格** 36 格。實測成本 2.5 s/step、40 s/格評測，預期約 2 小時，
   上限用滿約 4.2 小時。
4. 跑完把 `runs/` 全部拉回本機入庫並推上 origin，然後提醒使用者關掉機器。

## 現況：協議修好了，主網格沒跑過

E2–E23 的**所有防禦效果數字失效**，根因是攻擊端從來沒有 classifier-free
guidance（等同 w=1），而 SD v1.4 在 w=1 下幾乎不服從 prompt——那些實驗是在
防禦一個不存在的攻擊。保真度量測那一側的方法學結論全部存活。

四道約束的交集，每個門檻都由實測或人眼定錨：`tau_lpips` 掃 0.02/0.05/0.10、
`tau_acut`=0.04、`tau_chroma`=0.8、`beta_linf`=0。判準是**語意軸**
（SigLIP，通過對照；CLIP 沒通過），不是 `net_lpips`。

## 這個專案最重要的一條經驗

**「匹配失真」已經三次被證明是假的**：site S 買模糊（LPIPS 對模糊不收費）、
site C 買色調偏移（鈍化約束只看亮度，構造上對色度全盲）。每一次都是最佳化
找到了約束集的盲區。

所以：**每引入一個新的參數化，都必須先用等 LPIPS 多臂探針量出「現行約束集
對它的特徵失真收不收費」，再放進網格。** 不能等網格跑完才做。作法見
`scripts/p9_chroma_probe.py`，判準見 `docs/RESULTS_E28_chroma.md` §1。

同一條經驗的另一面：**不得憑文獻聲譽選指標**。ΔE00 是色差的國際標準，實測
在等 LPIPS 下完全分不出加性雜訊與可見色偏（2.46 vs 2.19）。NLPD 與 VIF 也是
同樣被推翻的前例。

## 開跑前務必知道的操作細節

- **綁定者診斷是常設步驟**。`scripts/e27_binding_check.py` 對任意 run 目錄
  運作。已經踩過四個假的綁定者（`max_dev` 兩次、防禦 margin、`L_fid` 裡係數
  為 1 的原始 lpips 項），每一個都會讓整批網格變成無效資料。
- **停止準則要開**（`--stop_on_plateau`）。固定步數讓不同格子被不同的東西
  綁住；`stop_reason` 為空代表用盡上限而非收斂，那一格不可用於跨 site 比較。
- **Lightning AI 的背景腳本不是 login shell**，必須用絕對路徑
  `/home/zeus/miniconda3/envs/cloudspace/bin/python3`，否則取到系統 python
  而缺 numpy。多層引號的 `for` 迴圈在 ssh 裡會壞掉，寫成腳本檔傳過去。
  環境重建約 5 分鐘，參考 `scripts/drivers/e27_calibration.sh`。
- **TF32 預設已關閉**（`src/utils/device.py`）。開著會讓同一份程式在 V100 與
  H100 上精度不同，BDIA 精確反演會從「好 5 個數量級」退化成「好 37 倍」。

## 不要重走的死路（全部有資料支撐，見 NEXT_SESSION §7）

- `net_lpips` 當防禦成功的判準
- 對抗性強健的感知度量（E-LPIPS / R-LPIPS / LipSim）——解的是相反的失效
- NLPD、VIF、GMSD、HaarPSI 當保真約束——量的是位移不是鈍化
- ΔE76 / ΔE00 / `dchroma` 當色度約束——量的是量值不是空間連貫性
- low rank（使用者 2026-07-30 排除）
- site L / E / W（A 族）——VAE 重建地板高於加性運作點
- site S（使用者 2026-08-01 決定不放進重跑；作為對照的價值最後再評估）
- 固定步數的網格

## 環境

- 本機 Python：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（不是 base）。
  torch 2.13.0+cu126、RTX 2050 4 GB。
- 測試：`python -m pytest -q`，基準 **247 passed / 1 skipped**。
- **本機跑得動分析、跑不動主網格**：SD v1.4 光 fp32 權重就 4.26 GB，實測
  512² 訓練 OOM；256² 可跑但每步 183 s 且結果不可用（門檻都在 512² 定的）。
  分析腳本（p1/p5/p7/p9/p10）在本機 GPU 上跑很快，該用就用。
- 遠端由使用者開啟並提供連線資訊。TWCC 已無法登入（帳號到期）。
- 密碼與 token 由使用者提供，**不得寫入任何入庫檔案**。

## 工作要求

- 一律用繁體中文回答，客觀學術語氣；程式碼關鍵字、函式名、套件名維持英文。
  **commit message 用英文。**
- 動手前先驗證假設（讀檔、跑指令），不要憑記憶猜 API。
- 修改論文方法要記 before/after：具體行號、原貌、原因。
  既有的五次修訂記在 `src/defense/objective.py` 的 docstring。
- 架構或實驗設計先提計劃討論再寫程式。環境問題直接修掉，不用寫進報告。
- 宣告完成前必須實際跑過並看到成功輸出；失敗就說失敗。
- 未經明確授權不得把分支併入 main（目前在 `claude/e20-fidelity-constraint`）。
- 禁止用 try/except 或條件跳過掩蓋症狀，要找根本原因。
- **指標之間出現矛盾時，把影像做成比對頁給使用者親眼判斷**，不要用數字自行
  定調。既有的比對頁在 `runs/p1_iso_lpips_probe/`、`runs/p5_semantic_axis/`、
  `runs/p7_attack_sanity/`、`runs/e27d_C_lr0.3/`、`runs/p10_chroma_ladder/`。
  τ_acut 與 τ_chroma 兩個門檻都是這樣定出來的。
- **推翻自己先前的判斷時，把錯誤的假設與推翻它的資料一起留在文件裡**，不要
  改寫成正確版本。這一階段留了三個這樣的紀錄：E22 的 p99=0.97 假設、
  `local_dchroma_dev` 的無效構造、以及 P9 判準第一版寫得太鬆。
- `runs/` 是唯一的證據來源，所有 CSV / JSON / log / PNG / HTML 一律入版控。
  改動 `.gitignore` 的 `runs/` 區塊時必須用 `git status --porcelain --ignored`
  確認沒有結果檔被排除。

最後一句：這個專案到目前為止的產出主要是**方法學**——四道約束怎麼定、綁定者
怎麼查、指標怎麼判別、什麼時候該相信人眼而不是數字。主網格是第一次在修好的
協議下量「非加性到底行不行」，結果是正是負都是可發表的，不要為了讓它是正的
而調參數。
