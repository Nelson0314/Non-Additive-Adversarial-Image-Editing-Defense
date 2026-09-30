# 重整執行計畫（使用者已裁定，由你執行）

依據你先前的診斷報告（本 thread 的回覆；全文也在 `.tmp/codex_audit/AUDIT.md`）。使用者要求**由你執行**，
協調端（Claude）只派工與驗收，每完成一項向使用者回報。每次只做被指定的那一項。

## 使用者對第 7 章問題的裁定

| # | 問題 | 裁定 |
|---|---|---|
| 1 | 已結束研究是否要能執行 | 不用。一律移入 `/archive`，只保存原狀 |
| 2 | 固定版本共用套件＋vendor 快照 | 照你的建議 |
| 3 | 共用套件由誰維護 | 照你的建議（單一責任範圍） |
| 4 | 根層結構 | 主要目錄為 `/color`、`/style`、`/baseline`、`/docs`（留空，之後寫論文），**不要有 lab 空間**。另需一個管理暫時性嘗試的好方法：使用者不想保留失敗的嘗試 |
| 5 | 歷史 CSV 是否只搬檔 | **一律改名，根除後患**：CSV 內的路徑欄與識別值也依新命名改寫（列數、鍵集合、數值欄須與基準逐一相同） |
| 6 | lab 遠端數值全部保全 | 照你的建議 |
| 7 | color 另外三個分支（`color_xattn`、`color_simple_dayn`、`color_lut3d_dayn`） | 移除 |
| 8 | 主表 1,408 列幾何分區欄 | 照你的建議（修正） |
| 9 | 報告生成器 | 不需保留（`.claude/skills/defence-report/` 一併移除） |
| 10 | 舊 artifact | 完全不管。文件中不再列 artifact 網址（`/archive` 內原文不動） |
| 11 | 遠端一起重整 | 要 |
| 12 | 換行基準 | 照你的建議 |

## 協調端補充的決定（使用者可推翻）

- 共用正本放根目錄 `/core`（`immunization_core` 套件＋GPU 租約工具），三個專案各附 `vendor/` 快照與 lock。
- `/archive` 內部維持原狀，不改名、不改 CSV；「一律改名」適用於 `/core`、`/baseline`、`/color`、`/style`。
- `colour_curve_ours` 改名為 `color_curve`（與 `color` 是不同方法，不可合併）。
- 幾何分區欄更正後直接寫回表中，更正前的值以 git 歷史保存，STATUS 記 commit id。
- 暫時性嘗試：各專案 `trials/<名稱>/`（本機與遠端都不入版控）；成功則把需要的東西搬進 `src/`、`configs/`、`results/`
  後 commit，再刪 trial；失敗整個刪除，只在 `docs/TRIALS.md` 留一列（試了什麼、設定、關鍵數字、結論來源）。
  附 `trial.sh new|promote|drop`；`drop` 同時清遠端並要求先寫那一列。

## 目標結構

```
/core      共用正本（immunization_core：editors、metrics、purifiers、pipelines、artifacts、runtime；GPU 租約工具）
/baseline  原 anti-purification/main_table；baseline 攻擊實作放這裡
/color     原 lab 顏色線
/style     原 lab 風格指令線
/docs      空
/archive   anti-purification 其餘、frequency-phase、根 HANDOFF.md 與 COLOUR_LINE.md、migration 紀錄
```
每個專案：`README.md`、`STATUS.md`、`pyproject.toml`、鎖定依賴、`src/`、`scripts/`、`configs/`、`data/`、`results/`、
`tests/`、`docs/`、`vendor/`；不入版控：`artifacts/`、`runtime/`、`trials/`。

## 工作項目

| 項 | 內容 |
|---|---|
| 0 | 凍結與基準（**已完成**：其他 session 已凍結；`archive/migration/pre_migration_manifest.json` 於 commit `8bcaae0`，由 `archive/migration/snapshot_manifest.py` 產生） |
| 1 | 修可靠性缺陷（在原位置修，行為修正各自獨立 commit）：readout.sh 失敗仍印完成、queue_worker 完成判定、color_row_chain.sh 目錄存在即略過、FLUX／UltraEdit 續跑鍵、style `feasible`、metrics_union 吞例外、run_on_card 取卡原子性、style_prompt_round 預設 CAP=6、`io.py` 尺寸判定、style_prompt_readout 空結果崩潰 |
| 2 | 移除：color 三個分支、`defence-report` 技能與模板、活動文件中的 artifact 網址 |
| 3 | 建 `/core`：抽出活動專案 import 閉包的模組；三份編輯／淨化／讀數副本先做行為差異表再合併（保留 `purified_mask()`）；拼法統一（optimize、defense、`IMMUNIZATION_ALLOW_TF32` 等）；相關測試隨行 |
| 4 | 建 `/baseline`：main_table 逐檔改名（你的 4.3、4.4、4.6）、資料快照、vendor、移除跨目錄搜尋 |
| 5 | 建 `/color`、`/style`：拆 lab、解除互相 import、遠端 `lab/runs/` 的數值 CSV 收進版控 |
| 6 | 建 `/archive`：搬入其餘內容，內部不動 |
| 7 | CSV 改寫：路徑欄與識別值依對照改寫；逐表驗證列數、鍵集合、數值欄與基準一致 |
| 8 | 程式風格與文件：依你的 5.1、5.2、5.5；建立 trials 機制 |
| 9 | 環境與自足驗證：pyproject、鎖定依賴、各專案複製到獨立目錄後可 import／`--help`／pytest；換行正規化並比對雜湊 |
| 10 | 遠端重整：租約與行程檢查、搬 `runs/` 影像、改寫家目錄腳本、dry-run 後切換 |
| 11 | 更新 Claude 記憶的路徑與適用範圍 |
| 12 | 幾何分區欄更正（遠端 1 張 GPU） |
| 13 | 收尾：根 README、根 CLAUDE.md（共通規則）、各專案 STATUS、殘留舊名查核 |
| 14 | （協調端）全部完成後，向 `baseline`、`color`、`style` 三個 session 發送完整注意事項：新結構、各自入口與管轄、文件與程式規範、trials 機制、GPU 租約工具、遠端新版面、解除凍結 |

## 執行規則

- 只做被指定的項目；做完即停，最終回覆列出：改了什麼（檔案與 commit）、如何驗證、驗證結果、未完成或需裁定的事。
- commit message 用英文；機械搬檔、行為修正、文件更新分開 commit。不 push。
- 不動未追蹤的 `1-s2.0-S0925231226019892-main.pdf`；不刪除任何未追蹤或 ignore 的資料，除非該項明文要求。
- Windows 寫出的 `.sh` 必須是 LF。
- GPU 卡數：全局可用卡由使用者逐次授權；未說明或說明不清時預設全局合計上限 6 張。本工作本身不跑 GPU。遠端（`ssh -p 10101|10102 nelson0314@server.basiclab.lab.nycu.edu.tw`）只在指定項目使用；若你的沙箱無法連線，
  把要在遠端執行的完整指令寫成腳本放 `.tmp/codex_audit/remote/` 並停下，由協調端代跑。
- 不設研究判準、不改科學協定（LPIPS 後端、resize、遮罩、seed、precision）。
- 書面用語依 `C:\Users\nelso\.claude\CLAUDE.md`。
