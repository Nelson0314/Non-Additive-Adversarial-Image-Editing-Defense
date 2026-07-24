# stage2 摘要（實測值）

- drop = (clean − purified) / clean，逐指標；lpips 之 drop 越大代表淨化越有效（防禦被移除越多）。方向定義見 METRIC_HIGHER_IS_BETTER；drop_valid=False 之列基準為負值，比值方向須個別解讀。
- 核心比較：加性（pg_*）vs 非加性（advdiff/apa/hybrid）之 drop。待驗證假設（非預設結論）：非加性之 drop 較小。

| purify | 加性 mean drop_lpips | 非加性 mean drop_lpips |
|---|---|---|
| jpeg_q90 | 0.1919 | 0.0270 |
| jpeg_q80 | 0.2990 | 0.0282 |
| jpeg_q65 | 0.4136 | 0.0155 |
| jpeg_q50 | 0.4813 | -0.0030 |
| blur_s0.5 | 0.0768 | 0.0253 |
| blur_s1.0 | 0.6350 | 0.0525 |
| blur_s1.5 | 0.7604 | 0.0522 |
| crop_r0.1 | 0.1484 | -0.0957 |
| crop_r0.2 | 0.1378 | -0.2126 |
| crop_r0.3 | 0.1205 | -0.2821 |
| advclean_bf | 0.3282 | -0.0337 |
| advclean_bfgf | 0.3031 | 0.0112 |
