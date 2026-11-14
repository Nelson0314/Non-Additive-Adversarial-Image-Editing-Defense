# 人像照片調查（第二版）

20 張照片，全部以「經濾鏡調色的人像」呈現，受試者想像為自己的照片作答兩題（1–5 分）：`share`（願意分享到社群平台）、`normal`（看起來像一般人發文的樣子）。開始前只問色覺異常，不收年齡與影像工作；題序依受試者隨機，附進度條，結束頁只顯示致謝。

| 條件 | 張數 | 來源 |
|---|---|---|
| ImmuLUT 防禦圖 | 18 | crosseditor100 中臉框 CIEDE2000 最小的 36 張，以人工看圖排除背景或衣物偏粉紫、局部色塊者後取 18 張：woman_87、man_19、man_34、woman_28、woman_95、woman_06、woman_43、woman_03、woman_30、woman_15、man_07、woman_68、woman_70、woman_96、woman_77、woman_89、woman_88、woman_41 |
| 常見濾鏡 | 2 | 第一版的暖色（woman_16）與褪色（woman_59），與上列人像不重複 |

條件對應在 `manifest.json`（含每張的防禦圖 LPIPS），影像由遠端 `user_study_make_site_v2.py` 產生（主種子 20260812 打亂檔名）。第一版（5 題、原圖／濾鏡／ImmuLUT 各條件）保存在 `pilot/`；依 LPIPS 區間挑選 18 張的版本（2 人作答）保存在 `lpips_bins/`。此版影像由遠端 `user_study_make_site_v3.py` 產生。

## 執行

```
python docs/user_study/site/run_survey_server.py --port 8000
```

- 同一個區域網路內的受試者開 `http://<本機 IP>:8000/`（本機 IP 以 `ipconfig` 查 IPv4 位址）；Windows 防火牆第一次執行時會詢問是否允許。
- 不在同一網路的受試者需要對外的通道（例如 Cloudflare Tunnel），由使用者自行決定與設定。
- 作答逐題寫入 `responses.jsonl`（已列入 `.gitignore`），每列一筆：`profile`（年齡區間、是否從事影像工作、色覺、題序）、`answer`（照片、五個分數、作答秒數）、`finish`。受試者中途離開時，已作答的題目仍保留。
