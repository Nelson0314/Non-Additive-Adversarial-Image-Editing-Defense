# 交付：inpainting 批次（EXP-ip4）

寫於 2026-08-09 22:5x。接手者請整篇讀完再動手——**這批還不能開跑**，
下方 §4 列出四個未解事項，其中兩個會直接讓段 0 中止。

---

## 1. 一句話說明這批要做什麼

在 inpainting 威脅模型下重跑第三階段的配置。條件與 EXP-s3a／s3t25／s3t30
逐字相同（apa、Ra、photoguard_c、mist、dia_r），換掉的是威脅模型：
9 通道 inpainting 權重、攻擊方帶一張遮罩、攻擊 prompt 改用 `prompts[1]`。

舊的 ip1／ip2／ip3 有 DEF-011（遮罩與式 (4) 的 M 重疊），**結構上量不到
防禦效果**，那三批作廢。本批是重做。

## 2. 使用者已裁決的事項（不要再改）

| 項目 | 裁決 | 出處 |
|---|---|---|
| 遮罩來源 | **人工繪製**，不由模型的 cross-attention 產生 | DEC-010 |
| 遮罩與 c_a 的關係 | c_a 在**遮罩外**（Lo Figure 3） | DEC-010 |
| 遮罩畫法 | 貼合物件輪廓，**不是方框** | 2026-08-09 |
| 實際作法 | 使用者描**主體**輪廓，程式翻面成「主體之外」 | 2026-08-09 |
| 攻擊 prompt | `prompts[1]`（保留 c_a、改動別處），`--prompt-index 1` | DEC-010 |
| 本批影像 | **horse_00、man_00、bird_03** | 2026-08-09 |

影像是使用者親自看過總覽圖後挑的。三張的遮罩涵蓋率分別是
**0.645 / 0.501 / 0.691**，都在可用窗口 [0.15, 0.45] 之外；此事已於挑選前
明確告知，使用者仍選這三張，照辦即可，但**要寫進報告的適用範圍**。

## 3. 已經完成並驗證過的部分

- `scripts/draw_masks.py`：Tk 繪製工具（套索／多邊形／筆刷、Shift 擦除、
  復原、即時涵蓋率），另有三個非互動模式 `--overview`／`--invert`／`--recrop`。
  以合成事件實測過：套索畫圓 0.1992（解析值 0.1963）、多邊形三角形 0.1751
  （0.1717）、擦除與復原正確、存出嚴格二值 512²。
- 24 張主體輪廓已由使用者描完，並已翻面、裁切。
  - 遮罩：`data/lo_masks/<image_id>.png`（255 = 重畫區）
  - 主體原稿：`data/lo_masks/_subject/`（裁切後）與 `_subject/_full/`（裁切前）
  - 裁切前原圖：`data/lo_original/<類別>/`
  - 逐張紀錄：`data/lo_masks/recrop.csv`、總覽圖 `data/lo_masks/overview.png`
  - **`data/lo_aligned/<類別>/*.png` 已被裁切後的版本覆蓋。** 這是資料集本身
    的改動，要入版控；原圖在 `data/lo_original/` 可還原。
- 載入端 `src/data/masks.py::load_drawn_mask`：缺檔／空遮罩／填滿整張都拋出，
  不落回自動產生的遮罩。
- 遮罩內容進 `config_hash`（`masks_digest`），改一張遮罩即全部格重跑。
- 不相交斷言 `src/models/attention.py::assert_masks_disjoint`，在
  `optimize.py` 對**真正進損失的 M** 斷言（見 §4-A：它很可能會拋）。
- 測試 **830 passed / 1 xfailed**（基準 817，新增 13）、`code_health.py` 0 項。

### 3.1 交付前查到並修掉的三個缺陷

1. `recrop` 只更新原圖與主體遮罩，**沒有重寫翻面後的攻擊遮罩**——
   `masks/*.png` 會留著裁切前的形狀疊在新圖上錯位，而那正是實驗讀進去的檔案。
2. `masks_digest` 用 `glob("*.png")`，把 `overview.png` 也算進雜湊；
   重產一次總覽圖就會靜默改掉每一格的 `config_hash`。已改為 `mask_files()`
   排除，並加測試 `test_總覽圖不得進遮罩雜湊`。
3. 遮罩目錄原本放在 `data/lo_aligned/masks/`，`load_lo_aligned` 有一道
   「未宣告卻含 PNG 的子目錄一律拒絕」的檢查（擋的是忘了宣告類別），
   **會在段 0 直接 KeyError**。已把遮罩與備份搬到 `data/lo_masks/`、
   `data/lo_original/`，並加斷言禁止路徑放回資料集內。

---

## 4. 未解事項——**開跑前必須處理**

### 4-A（會中止）不相交斷言很可能在段 0 拋出

`optimize.py` 會對 c_a 的注意力區 M 與遮罩斷言不相交。遮罩是「主體之外」，
M 理應落在主體上，但：

- M 定義在 **64×64** 的注意力格點上（512² 下一格 = 8×8 像素），比對用
  max-pool，**一個像素落進去就算重疊**；
- 翻面時只留了 `guard=13` 像素的保護帶，約 1.6 格；
- c_a 的注意力**不保證只落在物件上**，可能外溢到背景。

本機沒有模型，算不出 M，所以這件事**在遠端段 0 才會知道**。
建議先跑一支只做「載入遮罩 → 算 M → 斷言」的預檢（單卡、幾分鐘），
不要直接開整批 2028 格。若真的重疊，處置是提高 `--guard` 重跑
`draw_masks.py --invert --guard 21`（**會改 digest，全部格重算**），
而不是放寬斷言。

### 4-B（會中止）τ_train 尚未決定

`run_stage` 的 `--tau-train` 是必要輸入，且會導出 `tau_acut`（0.8τ）與
`tau_chroma`（16τ）。DEC-010 明訂**不可沿用 img2img 的 0.25／0.30**——
那些工作點是在 img2img 上由人眼定的，inpainting 的失真型態不同。

正解是先產 x_def 的 τ 掃描預覽讓使用者挑（要 GPU，可排在段 0 之前）。
**這是使用者的決定，不要自己填一個值。**

### 4-C（會排擠）目前沒有我們的空閒卡

22:32 實測：GPU 0/4/5 正在跑 s3t25 的段 1–3（eval 450/575、450/575、
350/575，`failed=0`），且 `wacv-chain-s3t30` 會在 s3t25 成功後**自動接手
整套流程**（段 0 約 1.1 h + 段 1–3 約 4.9 h），預計 05:20 才全部結束。
GPU 1/2/3/6 是 jayson、7 是 johnlee，**不要碰**。

使用者說「目前有空卡」，但那幾張低記憶體的卡是別人的。inpainting 批次要
三張卡跑約 5 小時，與 s3t30 直接衝突。開跑前先確認 s3t30 的狀態，
或與使用者確認要不要讓它插隊。

### 4-D（設計）三張圖的涵蓋率都偏高，且窗口的意義已改變

翻面後 `涵蓋率 = 1 − 主體佔比 − 保護帶`，所以**主體愈小、遮罩愈大**。
窗口 [0.15, 0.45] 是在「遮罩＝物件」的舊定義下量出來的，現在遮罩是
「整個背景」，同一組數字的意義不同，報告裡不可直接沿用舊判語。

裁切已把 24 張逼到極限（放大 1.06–2.26×，上限 3.0 未觸頂），仍有 16 張在
窗口外。原因是幾何上的：主體在其外接框內只佔 0.19–0.49（鳥最細長），
裁切改變不了這個比例。已向使用者說明，使用者選了其中三張。

---

## 5. 開跑指令（**4-A／4-B 解決後才可用**）

`scripts/shard.sh` 的 `ip*` profile 已更新：

```
MODEL="--model runwayml/stable-diffusion-inpainting --wrapper sd_inpaint --resolution 512"
MASK="--masks data/lo_masks"
GRID="--conditions apa Ra photoguard_c mist dia_r --tau-train <未定>"
```

還需要補上 `--prompt-index 1` 與 `--attn-mask-tau 0.5`、`--attn-timesteps 2`
（DEC-011），並確認 profile **沒有** `--strength`（inpainting 不接受，
`run_stage` 會擋）。`--warp-mask-gate` 已移除：閘只作用於 site warp 的三個
條件，而它們已依 DEC-005 移出格點。

上機順序固定為

```
source ~/env.sh && cd ~/WACV-s3 && ...
```

`env.sh` 最後一行會 `cd $HOME/WACV`，那是另一個 session 的工作樹
（分支 `claude/e20-fidelity-constraint`），順序寫反會跑到對方的程式碼，
且只有在旗標剛好不相容時才會報錯（DEF-012）。**不要動 `~/WACV`。**

## 6. 尚未 commit

本輪全部改動都還在工作區，`scripts/draw_masks.py`、`data/lo_masks/`、
`data/lo_original/` 是未追蹤檔。`data/lo_aligned/` 底下 24 張圖已被覆蓋。
入版控前用 `git status --porcelain --ignored` 確認沒有結果檔被排除
（`.gitignore` 的 `runs/` 區塊有前科，見 commit `1942e38`）。
