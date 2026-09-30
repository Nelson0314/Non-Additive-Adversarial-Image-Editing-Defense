# 風格轉換線（SPA 移植）

依據 Wang et al.,「Style-controllable adversarial example generation via image editing and prompt embedding
optimization」（Neurocomputing 702, 134591, 2026；DOI 10.1016/j.neucom.2026.134591，出處見 `references/README.md`）。原方法凍結 InstructPix2Pix、最佳化指令的文字
embedding，使風格編輯後的影像騙過分類器。本線改為免疫：防禦圖由 ip2p 依防禦方的風格指令產生，目標是使攻擊端
ip2p 的後續編輯失效。

規則：訓練不得使用任何編輯指令（評估指令與自選替代指令皆不可），文字只可用空字串或類別詞；不可加性雜訊；
主體身分與內容保持，衣服與背景可以變色；只用主種子。

## 攻擊端協定

- 攻擊方以 InstructPix2Pix（`timbrooks/instruct-pix2pix`，管線預設 scheduler）依文字指令編輯：50 步、s_t 7.5、s_i 1.8、
  主種子 20260812（`immunization_core.pipelines.editing` 的預設值，經 `immunization_style.cli.run_edits` 執行）。
- 指令：`data/portraits/prompts.yaml` 的 `edits.ip2p` 四句（墨鏡、警察制服、安全帽、領結）；防禦方在訓練中看不到這些指令。
- 資料：`data/portraits/`，8 張 512² 人像（`man_00..03`、`woman_00..03`），附主體遮罩（`masks/`，白為重繪區）。
- 未防禦分母：同一協定下原圖的編輯，位於 `artifacts/undefended_edits/ip2p_si18`。
- 讀數：`cli/measure_style_prompt_edits.py`，定義見下節「程式」。LPIPS 為 `piq.LPIPS()`，主體與背景分區使用
  `immunization_core.pipelines.masks.subject_mask()`。

## 程式

| 檔案 | 內容 |
|---|---|
| `immunization_style.method`、`cli/generate_style_prompt_defenses.py` | 生成器 G、載體、目標、限制與求解（選項見 `--help`） |
| `cli/measure_style_prompt_edits.py` | 讀數：編輯結果 LPIPS（對未防禦編輯、對風格參考圖的編輯）、編輯前後改變量 LPIPS，全圖／主體／背景 |
| `scripts/run_style_prompt_jobs.sh`、`cli/evaluate_job_outputs.py` | 一組實驗的排程：工作清單、經 `vendor/scripts/run_with_gpu_lease.sh` 取卡、防禦圖通過驗收即送主種子編輯，全部工作結束後以名為 `ref` 的工作（`--lr 0 --updates 1`）為參照計算讀數；各階段記錄結束碼並驗收鍵集合與輸出檔，既有輸出須與 `job.spec` 的設定相同才沿用，任一階段失敗即以結束碼 1 結束 |

## 設計

- 生成器 G：同一個 ip2p，20 步；前 5 步不回傳梯度，後 15 步回傳。`--sampler ddim`（預設）或 `ddpm`（論文附錄 B）。
  s_t 7.5、s_i 2.0（s_i 1.5 會換臉）。風格參考圖 x_ref ＝ 不做防禦訓練的 G 輸出（δ ＝ 0、z_off ＝ 0），為對照組。
- 載體：`prompt`（指令 token embedding 位移 δ；`--prompt-scope full` 為整段 77 token）、`latent`（暖身後 latent 位移 z_off）、`prompt_latent`。
- 目標（`--objective`）：`enc_gray`（E(y) 推向灰圖 latent）、`attn`（自注意力偏離）、`xattn`（類別詞交叉注意力比值，最小化）、
  `chaos`（攻擊端以類別詞為指令的 10 步編輯輸出與輸入的灰階 LPIPS，最大化）、`classifier`（論文原損失
  exp(κ·tanh(m/κ))，κ 9，ResNet-50，標籤取原圖 top-1）、`free`。
- 限制：FaceNet 身分下限、臉部暖色上限、`--struct-cap` 灰階 LPIPS 結構上限（對 x_ref）。停滯規則：連續 `--patience` 次驗證改善不到 1% 即 lr/4，最多 `--max-decays` 次。

## 結果

本專案不保存結果。已刪除的實驗設定與關鍵數字記錄於 `docs/TRIALS.md`，數值 CSV 可由該表所列的 commit 取回。
