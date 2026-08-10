# 交接：APA+ 的 A 段（重建下限）

本檔是給**新 session** 的完整交接。設計依據是 `docs/DECISIONS.md` 的 **DEC-016**，
事實依據是 `docs/FINDINGS.md` 的 **FND-016 ~ FND-019**。兩者都必須先讀完。

使用者要求：**用最快的速度先看到 A 段的結果，而且要真實比對圖。**
B 段（約束）與 C 段（損失）**在 A 段驗收之前不要動**。

---

## 1. 一句話說明要做什麼

把 APA 的「階段一」從裝錯的地方（UNet 的 LoRA）搬到真正產生重建誤差的地方
（VAE 的 latent 與解碼器），把 `φ=0` 的重建下限壓下來，然後**產比對圖給使用者看**。

## 2. 為什麼（不要跳過）

走生成路徑的條件（`apa`、`Ra`、`N3`）產生防禦圖的方式是
`decode(BDIA 反演(encode(x)) 之後注入 φ)`。即使 `φ=0`，`decode(encode(x))` 也不等於
`x`——這就是重建下限，實測 LPIPS 0.128–0.158、DISTS 0.035–0.049。

現行的階段一（`src/defense/optimize.py::align`）想壓的就是它，但**壓不動**：

- 它把四道 hinge 的門檻設成該影像自己的重建下限（`recon_floor_thresholds`），
  所以 `φ=0` 一開始就滿足約束，`align_loss` 從第 1 步起恆為 0
- 更根本的是下限來自 **VAE** 的 encode/decode，而 LoRA 在 **UNet** 裡，碰不到它
- APA 原論文用 DDIM 反演，他們的階段一補的是**反演誤差**；本專案用 BDIA
  精確反演（fp32 下與純 VAE 來回逐位相同），那一半誤差本來就不存在
- 實測：200 步只讓 LPIPS 從 0.15806 移到 0.15768，每格卻花 940 秒

證據：`runs/ip20_horse_00/apa/horse_00/align.csv`（若已回收）或遠端同路徑。

## 3. 要做的兩件事

### A1 · latent 對齊

不要直接拿 `encode(x)` 當生成路徑的起點，改為**解一個 `z*`** 使 `decode(z*)`
盡量等於 `x`。參數量與既有 latent 相同（SD v1.4/512² 是 4×64×64），不新增模組。

- 專案先前量過：LPIPS 0.1434 → **0.0760（−47%）**
  （`docs/archive/PRIOR_FINDINGS.md` §4.3 的表）
- 同一張表也記著「不是欠調參」：4 倍步數、1/4 學習率，落點都在 0.075 附近

### A2 · 解碼器逐圖微調

在 VAE 解碼器上開一組**小參數**對這一張影像過擬合。

- **先只開 GroupNorm 的 affine（weight/bias）與各層 conv 的 bias**，數千個參數。
  不要一開始就全參數微調：全解碼器約 5×10⁷ 參數，逐圖存下來是 198 MB／張，
  `runs/` 放不下（`.gitignore` 的 runs 區塊允許 `*.pt`，但體積不可接受）
- 專案先前量過的是「換一個**通用**的非對稱 decoder」只有 −10%；
  **逐圖過擬合沒量過**，預期強很多。A1+A2 疊加的舊量測是 **0.0716（−50%）**

### 兩個硬性要求

1. **A2 必須有硬停止條件。** 解碼器若把原圖背得太熟，會對 latent 的擾動變得
   遲鈍——你在 latent 上加東西它照樣吐原圖，防禦就沒有管道表達。
   故達到下限目標即停，不跑到收斂。停止步數與當時的下限值必須落盤。
2. **不得用 try/except 或條件跳過掩蓋症狀**（專案規則）。達不到就如實拋出並記錄。

## 4. 驗收方式（使用者要看的東西）

**不必跑 eval。** 只要量新的 `φ=0` 下限並產出**逐圖比對頁**：

| 欄 | 內容 |
|---|---|
| 1 | 原圖 |
| 2 | 舊下限（現行 `decode(encode(x))`） |
| 3 | 新下限（A1 之後） |
| 4 | 新下限（A1+A2 之後） |
| 5 | 差分放大圖（新舊各一） |

每一格都要標 LPIPS／DISTS／PSNR／銳利度比。影像用
`horse_00 horse_03 woman_03`（img2img）與 `horse_00 man_00 bird_03`（inpainting）。

比對頁做成單一自足的 HTML（影像以 base64 內嵌），參考
`scripts/tau_preview.py::render` 與 `scripts/report_t25.py` 的既有樣式。

**機時約 10 分鐘**（只有前向與一段短最佳化，沒有編輯鏈）。

## 5. 環境

- 遠端：`ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw`（basic-1，8×RTX 3090）
  與 `-p 10102`（basic-2，8 張，其中 5/6/7 通常空著）。**兩台共用同一個 NFS home
  與同一個 `~/WACV-s3` 工作樹。**
- 每次執行前 `source ~/env.sh`，但注意它最後一行會 `cd $HOME/WACV`（**另一個
  session 的樹**），所以 source 之後要自己 `cd ~/WACV-s3`
- 產物寫在 `~/wacv_runs/`（版控範圍之外），回收後才進 `runs/`
- 本機 Python：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`（不是 base）
- 測試基準：`python -m pytest -q` 應為 837 passed / 1 xfailed
- `python scripts/code_health.py` 應為 0 項

**不要碰**：GPU 1/2/3/6 是 jayson 的、7 是 johnlee 的（basic-2 上 0–4 有人在用，
5/6/7 空）。開跑前用 `nvidia-smi` 確認。

## 6. 目前的狀態（都已停止）

| 批次 | 狀態 |
|---|---|
| `s3t20`（img2img，Δ=0.04） | **完整跑完**，已合併、已出判定層。`runs/s3t20_merged/`、`runs/s3t20_protocols/` |
| `ip20`（inpainting，Δ=0.04） | 段 3 停在 eval 300–350/475。逐格紀錄完整，可用同一命令續跑（約 26 分鐘）。`runs/ip20_*/` |
| Δ 掃描 | 完成。`runs/tp_sweep2/`、`runs/tpi_sweep/` |

遠端 tmux 已全部 kill，沒有任何 session 在跑。

## 7. 使用者已經定案、不要重新討論的事

- **Δ = 0.04（相對 DISTS）可接受**。原話：「除了女人照片那張之外，其他的照片
  失真都在可接受範圍，並且其實各個 delta DIST 看起來的差距沒有到真的很大」
- woman_03 在 Δ=0.04 的銳利度比 0.664、Δ=0.02 仍只有 0.730，**降 Δ 救不了它**
- 預算軸用相對 DISTS（DEC-015），不要退回絕對 LPIPS
- 式 (5) 對整個 M 取，不做區域限制（DEC-014）
- 判準以人眼為主、數值指標為輔；`compare.html` 是主要產出物

## 8. 專案規則（違反會被退回）

- 一律**繁體中文**回答，客觀學術語氣；程式碼關鍵字、函式名、套件名維持英文。
  **commit message 用英文。**
- 動手前先驗證假設（讀檔、跑指令），不要憑記憶猜 API
- 修改論文方法要記 before/after：具體行號、原貌、原因
- 架構或實驗設計需先提計劃討論再寫程式
- 宣告完成前必須實際跑過並看到成功輸出；失敗就直說失敗
- **未經明確授權不得把分支併入 main**（目前在 `claude/stage3-apa-attn`）
- 禁止用 try/except 或條件跳過來掩蓋症狀
- 密碼與 GitHub token **不得寫入任何入庫檔案**

## 9. B 段與 C 段（A 驗收後才做，先不要碰）

摘要見 DEC-016。重點：

- B1 訓練與評測用同一個 Δ（現在訓練綁 τ_LPIPS=0.20、評測綁 Δ_DISTS=0.04，差六倍）
- B2 新增最差區塊 hinge（`local_acutance_worst` 改為可微並進損失）
- C1 `L = L_attn + λ·L_targeted`
- C2 `attn_timesteps` 2 → 4，作法是逐 pair 反傳再累加梯度
- C3 把 `N3` 拉回格點，構成 Ra／N3／apa／apa_plus 的損失消融
- 連帶：`loss_params` 改為逐條件產生，使加性 baseline 不會因為我們改損失而重跑
  （既有批次的雜湊必須逐位不變，要有測試釘住）
