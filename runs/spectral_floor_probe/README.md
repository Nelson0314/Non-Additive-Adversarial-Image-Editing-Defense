# 頻譜加性下限探針的逐條件平均

**這個方向已被否決**（`docs/DECISIONS.md` 的「頻譜加性項不做」）。本目錄只保留
量測數字，作為那條裁決的證據——裁決是在看過這些數字之後重申的，不是憑推測。

**程式不在版控裡。** 依使用者裁定，探針是一次性的。要重建的話，改動只有一處：
在 `PhaseResidual` 的前向裡把

```
spec' = |spec|·exp(g·gate) · exp(i(φ + θ·gate))
```

換成

```
spec' = (|spec|·exp(g·gate) + a·q(ω)·scale) · exp(i(φ + θ·gate))
```

`a` 逐（區塊, 頻格）可學、夾在 `[-1, 1]`，與 `theta`／`gain` 走同一條 PGD；
`q` 是 `src/residual/perceptual_weight.py` 的 `jpeg_luma` 權重；`scale` 即
`floor_scale` 欄。**加性項只乘徑向帶通遮罩、不乘紋理閘**——它存在的理由就是
要進到紋理閘擋掉的平坦區，乘了紋理閘就什麼也沒做。

`blocked_clip` 是 `clip_sim < 0.8445` 的格數（判準與門檻見
`docs/EVALUATION.md`）。影像集是 13 張人眼確認服從的影像，
清單在 `runs/obedience_audit/recognisability_verdict.csv`。
