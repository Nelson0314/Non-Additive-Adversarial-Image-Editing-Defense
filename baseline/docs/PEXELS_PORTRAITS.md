# Pexels 人像資料組

`data/pexels_portraits/`：男 48 張、女 49 張，512² RGB PNG，構圖與 `data/portraits/` 同型（單人、近正面、半身或全身）。
用於在較大的影像集上重跑十一個外部條件的原生預算評測（ip2p 場景），入口為 `scripts/evaluate_pexels_portraits.sh`。

## 來源與授權

- 中繼資料：Hugging Face `gaunernst/pexels-portrait`（`pexels_portrait_part1.csv.gz`，9,853 列，Pexels API 欄位，已以 YOLOv8-Face 去除無臉影像）。
- 影像：`images.pexels.com` 的原圖，參數 `auto=compress&cs=tinysrgb&w=1200`，依 Pexels License 使用。
- `provenance.json` 逐張記錄 Pexels ID、頁面與影像網址、攝影師、alt 文字、原尺寸、臉框、裁切框與輸出 PNG 的 SHA-256。

## 選取

1. alt 文字只含男性或女性其中一類詞，且不含兒童、群體、黑白、遮擋、藝術化描述等排除詞：5,763 張。
2. MTCNN（`facenet_pytorch`，與 `immunization_core.metrics.identity` 相同偵測器）：機率 ≥ 0.90 的臉恰一張且其機率 ≥ 0.98；
   臉框高度 / 短邊介於 0.07 與 0.50；鼻尖對兩眼中點的水平偏移 ≤ 0.25 倍眼距；Lab 平均 |a*| + |b*| ≥ 6；
   臉框完整落在裁切內：2,486 張。
3. CLIP ViT-L/14 零樣本 `a photo of a man` / `a photo of a woman` 的機率 ≥ 0.8 且與 alt 的類別一致：2,460 張。
4. 依裁切後影像與 `data/portraits` 八張的 CLIP 影像餘弦平均值排序，同一攝影師至多兩張，各類別取前 50 張與 30 張備用。
5. 目視剔除臉被遮擋、低頭或側臉、人物過小、臉部彩繪、同一場拍攝的近似重複等影像，以備用影像遞補；
   第二次目視再剔除男 7 張、女 6 張（其中男 5 張、女 5 張以備用影像遞補），編號依序重排。

裁切沿用 `archive/anti-purification/scripts/crop_portraits.py`：邊長 min(H, W)，中心為臉框中心往下 0.18 倍邊長，
超界夾回，LANCZOS 縮至 512²。

## 遮罩

`masks/` 由 `immunization_baseline.cli.generate_subject_masks` 產生（CLIPSeg `CIDAS/clipseg-rd64-refined`，
文字為類別的 `content`，門檻 0.30，膨脹 4 px，白 = 重繪）。同一程式在 `data/portraits` 上的輸出與已入庫的八張遮罩逐位元相同。
主體面積 0.168–0.748。`masks/overview/` 中可見的分割誤差：`man_00`（窗框併入主體）、`man_36`（軀幹有缺口）、
`woman_03`（帽頂未含）、`woman_10` 與 `woman_46`（頭髮上緣未含）。遮罩只影響主體／背景分區欄與 `diffvax`。

## 評測協定

與主表相同：`timbrooks/instruct-pix2pix`，`s_t` 7.5、`s_i` 1.8，種子 20260812，50 步，512²，四條 FaceLock 指令；
七道淨化同 `immunization_core.purifiers.protocol`；LPIPS 為 `piq.LPIPS()`。只跑 ip2p 場景，不含 `color` 條件。

| 產物 | 位置 |
|---|---|
| 防禦圖、編輯、淨化 | `artifacts/pexels_portraits/`（子樹與主表 `artifacts/` 相同） |
| 逐條件讀數 | `results/pexels_portraits/defense_<條件>.csv`、`displacement_<條件>.csv`、`retention_<條件>.csv` |
| 執行狀態 | `runtime/pexels_portraits/*.done` |

依主表逐條件實測的求解時間，每張影像的 GPU 時間約為防禦生成 4.08 小時、ip2p 編輯（12 arm × 4 指令 × 未淨化與七道淨化，
每格 9–12 秒）1.0–1.4 小時，97 張單卡合計約 500 小時。條件依求解成本由低到高執行，`photoguard_c`、`photoguard_linf` 在最後。
