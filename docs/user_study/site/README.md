# 社群貼文照片調查（試測版）

`docs/USER_STUDY.md` 的單張評分設計之試測網站：20 張人像，每張評整體 1 題與細項 4 題（皆 1–5 分），題序依受試者隨機，附進度條，結束頁只顯示致謝。

## 影像

| 條件 | 張數 | 來源 |
|---|---|---|
| ImmuLUT 防禦圖 | 10 | crosseditor100 中位數附近 28 張（`docs/USER_STUDY.md` 第 2 節）以主種子 20260812 抽出 |
| 原圖 | 5 | crosseditor100 其餘影像，同一種子抽出 |
| 常見濾鏡 | 5 | 同上，各套一種濾鏡（暖色、冷色、褪色、高飽和、低飽和） |

每張影像只出現一次，檔名 `photo_01.jpg`–`photo_20.jpg` 已打亂，條件對應在 `manifest.json`（不在服務目錄內，受試者讀不到）。影像由遠端 `user_study_make_site_images.py` 產生，512×512、JPEG 品質 90。

## 執行

```
python docs/user_study/site/run_survey_server.py --port 8000
```

- 同一個區域網路內的受試者開 `http://<本機 IP>:8000/`（本機 IP 以 `ipconfig` 查 IPv4 位址）；Windows 防火牆第一次執行時會詢問是否允許。
- 不在同一網路的受試者需要對外的通道（例如 Cloudflare Tunnel），由使用者自行決定與設定。
- 作答逐題寫入 `responses.jsonl`（已列入 `.gitignore`），每列一筆：`profile`（年齡區間、是否從事影像工作、色覺、題序）、`answer`（照片、五個分數、作答秒數）、`finish`。受試者中途離開時，已作答的題目仍保留。
