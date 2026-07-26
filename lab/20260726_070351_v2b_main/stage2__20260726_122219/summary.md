# stage2 摘要（實測值）

- drop = (clean − purified) / clean，逐指標；lpips 之 drop 越大代表淨化越有效（防禦被移除越多）。方向定義見 METRIC_HIGHER_IS_BETTER；drop_valid=False 之列基準為負值，比值方向須個別解讀。
- 核心比較：加性（pg_*）vs 非加性（advdiff/apa/hybrid）之 drop。待驗證假設（非預設結論）：非加性之 drop 較小。

| purify | 加性 mean drop_lpips | 非加性 mean drop_lpips |
|---|---|---|
| jpeg_q90 | 0.1155 | -0.1048 |
| jpeg_q80 | 0.1901 | -0.1895 |
| jpeg_q65 | 0.2373 | -0.2703 |
| jpeg_q50 | 0.1929 | -0.3686 |
| blur_s0.5 | -0.0288 | -0.0820 |
| blur_s1.0 | -0.0972 | -0.5983 |
| blur_s1.5 | -0.2873 | -0.7029 |
| crop_r0.1 | -0.0918 | -0.2310 |
| crop_r0.2 | -0.0963 | -0.3794 |
| crop_r0.3 | -0.1093 | -0.3938 |
| advclean_bf | 0.0911 | -0.3852 |
| advclean_bfgf | 0.1493 | -0.3170 |
