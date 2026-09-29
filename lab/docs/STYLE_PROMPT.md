# 風格轉換線（SPA 移植）

依據 Wang et al.,「Style-controllable adversarial example generation via image editing and prompt embedding
optimization」（Neurocomputing 702, 134591, 2026；根目錄 PDF）。原方法凍結 InstructPix2Pix、最佳化指令的文字
embedding，使風格編輯後的影像騙過分類器。本線改為免疫：防禦圖由 ip2p 依防禦方的風格指令產生，目標是使攻擊端
ip2p 的後續編輯失效。協定與攻擊端同 `DESIGN.md` §1。

規則：訓練不得使用任何編輯指令（評估指令與自選替代指令皆不可），文字只可用空字串或類別詞；不可加性雜訊；
主體身分與內容保持，衣服與背景可以變色；只用主種子。

## 程式

| 檔案 | 內容 |
|---|---|
| `code/style_prompt_defence.py` | 生成器 G、載體、目標、限制與求解（選項見 `--help`） |
| `code/style_prompt_readout.py` | 讀數：編輯結果 LPIPS（對未防禦編輯、對風格參考圖的編輯）、編輯前後改變量 LPIPS，全圖／主體／背景 |
| `scripts/style_prompt_round.sh` | 一輪實驗的排程：工作清單、`flock` 取卡、防禦圖產出即送主種子編輯並重算讀數；名為 `ref` 的工作（`--lr 0 --updates 1`）產生對照 |

## 設計

- 生成器 G：同一個 ip2p，20 步；前 5 步不回傳梯度，後 15 步回傳。`--sampler ddim`（預設）或 `ddpm`（論文附錄 B）。
  s_t 7.5、s_i 2.0（s_i 1.5 會換臉）。風格參考圖 x_ref ＝ 不做防禦訓練的 G 輸出（δ ＝ 0、z_off ＝ 0），為對照組。
- 載體：`prompt`（指令 token embedding 位移 δ；`--prompt-scope full` 為整段 77 token）、`latent`（暖身後 latent 位移 z_off）、`prompt_latent`。
- 目標（`--objective`）：`enc_grey`（E(y) 推向灰圖 latent）、`attn`（自注意力偏離）、`xattn`（類別詞交叉注意力比值，最小化）、
  `chaos`（攻擊端以類別詞為指令的 10 步編輯輸出與輸入的灰階 LPIPS，最大化）、`classifier`（論文原損失
  exp(κ·tanh(m/κ))，κ 9，ResNet-50，標籤取原圖 top-1）、`free`。
- 限制：FaceNet 身分下限、臉部暖色上限、`--struct-cap` 灰階 LPIPS 結構上限（對 x_ref）。停滯規則：連續 `--patience` 次驗證改善不到 1% 即 lr/4，最多 `--max-decays` 次。

## 結果（man_01 等 3 張人像；主種子；看圖判定配件）

| 設定 | 訓練 | 目標值 | 防禦圖 | 配件未畫出 |
|---|---|---|---|---|
| enc_grey，prompt＋latent，無結構上限 | 收斂 | 1.00 → 0.21–0.32 | 整張灰霧（對 x_ref LPIPS 0.34） | 1／4（安全帽） |
| attn，prompt＋latent，無結構上限 | 300 步未收斂，中止 | 持續上升 | 背景改寫為幻覺場景 | 未評估 |
| xattn，結構上限 0.08（`r11`） | 收斂 | 1.00 → 0.85–0.96 | 完好 | 0／12 |
| chaos，結構上限 0.08（`r11`） | 2／3 收斂 | 偏離 ＋0.005–0.16 | 完好 | 1／12（安全帽） |
| 論文設定（DDPM、77 token、lr 0.1、軌跡 ×25、prompt 餘弦 ×1）＋ xattn／chaos，20 步 | 依論文 | xattn 0.98–0.99 | 同參考圖 | 0／48 |
| 同上，訓練至收斂（`r13`） | 收斂 | xattn 0.97–0.98；chaos ＋0.03–0.08 | chaos 一張臉部龜裂紋理 | 0／18（看圖） |
| 論文完全移植：classifier 損失，20 步（`cls`） | 成功：4／4 分類器邊界轉負 | 1.50 → −2.81 等 | 對 x_ref LPIPS 0.05–0.11 | 0／16 |

- 論文方法可移植並騙過分類器，但騙過分類器與阻止 ip2p 編輯之間沒有關聯；編輯前後改變量與 x_ref 持平。
- 不讀指令的內部目標（encoder、注意力、攻擊代理輸出）在防禦圖完好時只移動 2–15%；能推動的設定以破壞防禦圖外觀為代價。
- Codex 診斷：論文靠分類器決策邊界監督，換成內部距離後與編輯失敗失去關聯；G 的輸出落在 ip2p 自身的自然影像分布上。
- 未試：分類器損失接在攻擊端以類別詞為指令的輸出上。

## 資料位置（遠端 `~/image-immunization/lab/runs/`）

`style_prompt_r11{,_edit}`（xattn／chaos，cool grading）、`style_prompt_r13{,_edit}`（論文設定、至收斂）、
`style_prompt_cls_{p_noedit,p_snow}{,_edit}`（論文完全移植）；工作清單 `specs/style_prompt_cls_*.txt` 與各輪目錄內的 `jobs.spec`。
