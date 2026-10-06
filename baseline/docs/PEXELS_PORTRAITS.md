# Pexels 人像資料組

`data/pexels_portraits/`：男 110 張、女 199 張，共 309 張，512² RGB PNG，構圖與 `data/portraits/` 同型（單人、近正面、半身或全身）。
用於在較大的影像集上重跑外部條件的原生預算評測（ip2p 場景），入口為 `scripts/evaluate_pexels_portraits.sh`。

## 來源與授權

- 中繼資料：Hugging Face `gaunernst/pexels-portrait`（`pexels_portrait_part1.csv.gz`，9,853 列，Pexels API 欄位，已以 YOLOv8-Face 去除無臉影像）。
- 影像：`images.pexels.com` 的原圖，參數 `auto=compress&cs=tinysrgb&w=1200`，依 Pexels License 使用。
- `provenance.json` 逐張記錄 Pexels ID、頁面與影像網址、攝影師、alt 文字、原尺寸、臉框、裁切框與輸出 PNG 的 SHA-256。

## 選取

1. alt 文字只含男性或女性其中一類詞，且不含兒童、群體、黑白、遮擋、藝術化描述等排除詞：5,763 張。
2. MTCNN（`facenet_pytorch`，與 `immunization_core.metrics.identity` 相同偵測器）：機率 ≥ 0.90 的臉恰一張且其機率 ≥ 0.98；
   臉框高度 / 短邊介於 0.07 與 0.50；鼻尖對兩眼中點的水平偏移 ≤ 0.25 倍眼距；Lab 平均 |a*| + |b*| ≥ 6；
   臉框完整落在裁切內：2,486 張。
3. CLIP ViT-L/14 零樣本 `a photo of a man` / `a photo of a woman` 的機率 ≥ 0.8 且與 alt 的類別一致，且 FaceNet 取得到身分向量：
   2,459 張（男 407、女 2,052）。
4. 依裁切後影像與 `data/portraits` 八張的 CLIP 影像餘弦平均值排序，同一攝影師至多兩張；
   與同類別已選影像的 FaceNet 身分餘弦 ≥ 0.65 者略過（同一人或同一場拍攝）。
5. 目視剔除：臉被遮擋、低頭或側臉、閉眼、人物過小、臉部彩繪或戲服、兒童、赤裸上身、過暗。
6. 主體遮罩面積落在 [0.01, 0.80] 之外的影像不納入（3 張）。

影像序 0–96（`man_00`–`man_47`、`woman_00`–`woman_48`）為第一批，由各類別前 80 名候選目視選出。
影像序 97 起為擴充：男性候選 161 張（通過步驟 4 的全部男性）目視通過 62 張，女性前 430 名候選目視通過 153 張、其中 3 張因步驟 6 排除，
依 CLIP 相似度由高到低續編號。男性在來源中通過步驟 3 的只有 407 張，兩類別的張數因此不相等。

裁切沿用 `archive/anti-purification/scripts/crop_portraits.py`：邊長 min(H, W)，中心為臉框中心往下 0.18 倍邊長，
超界夾回，LANCZOS 縮至 512²。

## 遮罩

`masks/` 由 `immunization_baseline.cli.generate_subject_masks` 產生（CLIPSeg `CIDAS/clipseg-rd64-refined`，
文字為類別的 `content`，門檻 0.30，膨脹 4 px，白 = 重繪）。同一程式在 `data/portraits` 上的輸出與已入庫的八張遮罩逐位元相同。
309 張的主體面積為 0.014–0.792。`masks/overview/`（不入版控，由同一 CLI 重新產生）中，第一批可見的分割誤差：`man_00`（窗框併入主體）、`man_36`（軀幹有缺口）、
`woman_03`（帽頂未含）、`woman_10` 與 `woman_46`（頭髮上緣未含）。遮罩只影響主體／背景分區欄與 `diffvax`。

## 評測協定

與主表相同：`timbrooks/instruct-pix2pix`，`s_t` 7.5、`s_i` 1.8，種子 20260812，50 步，512²，四條 FaceLock 指令；
七道淨化同 `immunization_core.purifiers.protocol`；LPIPS 為 `piq.LPIPS()`。只跑 ip2p 場景，不含 `color` 條件。

條件（`scripts/evaluate_pexels_portraits.sh` 的預設）：`diffvax`、`photoguard_encoder`、`mist`、`dia_pt`、`sifm`、`dia_r`、
`dct_shield_y`、`dct_shield`、`dayn`、`danp`。`photoguard_encoder` 為 PhotoGuard 的 encoder attack
（官方 img2img notebook：L∞ 0.06／[-1,1]、1,000 步、步長由 0.02 線性衰減、隨機起點）；
主表的 `photoguard_c`、`photoguard_linf` 為 diffusion attack，每張求解約 6,400 秒，不在預設內。

## 影像組與產物

`image_sets.json` 把影像序切成影像組：`images_0000_0096` 為第一批 97 張，其後每 10 張一組（`images_0097_0106`…）。
每個影像組完成全部條件的防禦生成、編輯、七道淨化、量測與補充指標（FSIM、ΔE00、MSE 與七項無參考美學指標）後才進行下一組。

| 影像組 | 產物 | 狀態與讀數 |
|---|---|---|
| `images_0000_0096` | `artifacts/pexels_portraits/` | `runtime/pexels_portraits/*.done`、`runtime/pexels_portraits/results/` |
| 其餘 | `artifacts/pexels_portraits/sets/<影像組>/` | `runtime/pexels_portraits/sets/<影像組>/`（`*.done` 與 `results/`） |

各影像組的資料子集為 `runtime/pexels_portraits/data/<影像組>/`（指向本資料組的符號連結）。
入版控的讀數在 `results/pexels_portraits/`（`defense_<條件>.csv`、`displacement_<條件>.csv`、`retention_<條件>.csv`）。

執行：`ENV_FILE=~/env.sh bash scripts/evaluate_pexels_portraits.sh <卡號> <影像組> [<影像組>...]`；
中斷後以同一指令接續（完成標記與 `results_*.csv` 的既有列）。

## 第一批 97 張的既有讀數

單卡 111 小時（`runtime/pexels_portraits/pause.log`）：未防禦分母與 `diffvax`、`mist`、`dia_pt`、`sifm`、`dia_r`、`dct_shield_y`、
`dct_shield` 七個條件完成防禦、編輯、七道淨化與量測；`dayn` 有 73 張防禦圖（`results/pexels_portraits/defense_dayn.csv`）。
報告頁位於 `artifacts/pexels_portraits/report/index.html`（不入版控）。

## GPU 時間

每張影像的防禦生成時間（秒；`defense_<條件>.csv` 的 `total_seconds` 平均）：`diffvax` 0、`dia_pt` 69、`mist` 77、`sifm` 123、
`dia_r` 147、`dct_shield` 155、`dct_shield_y` 155、`dayn` 483（73 張），以上為第一批實測；`danp` 569、`photoguard_c` 6,414、
`photoguard_linf` 6,481 為主表 8 張的實測。ip2p 編輯為 64 個 arm（每個 388 格）的平均，每格 11.6 秒。

| 配置 | 每張 GPU 小時 | 100 張 | 200 張 | 500 張 | 2,000 張 |
|---|---|---|---|---|---|
| 11 條件（含兩個 photoguard diffusion attack），含七道淨化（每張 384 格） | 5.31 | 531 | 1,063 | 2,657 | 10,629 |
| 同上，只跑未淨化（每張 48 格） | 4.23 | 423 | 846 | 2,115 | 8,461 |
| 去掉兩個 photoguard diffusion attack，含七道淨化（每張 320 格） | 1.53 | 153 | 305 | 763 | 3,052 |
