# 下一階段：協議已就緒，主網格待跑

寫於 2026-08-01（E28 之後），取代同日稍早的版本。完整過程見
`docs/RESULTS_E25-E26.md`、`docs/RESULTS_E27_calibration.md`、
`docs/RESULTS_E28_chroma.md`。

---

## 1. 一句話交代現況

E25–E28 修好了協議。**E2–E23 的所有防禦效果數字失效**（攻擊端沒有
classifier-free guidance，等同 w=1，該設定下 SD v1.4 幾乎不服從 prompt），
但保真度量測那一側的方法學結論全部存活，並新增了第三道約束。

**主網格尚未跑過。** 設定已全部定案，剩下的是 GPU 機時。

---

## 2. 為什麼既有數字不能用

| 問題 | 影響 | 出處 |
|---|---|---|
| 攻擊端無 CFG（w=1） | E2–E23 是在防禦一個不存在的攻擊 | E26 §3 |
| `net_lpips` 量的是輸出移動了多少 | 726 格中語意失敗 0 格 | E25 §1 |
| 固定步數 | 不同格子被不同的東西綁住 | E21–E23 §5.4 |
| 匹配失真三次是假的 | site S 買模糊、site C 買色調偏移 | E20、E27 §4 |

---

## 3. 存活的結論

全部與保真度量測有關，不經過 SDEdit，故不受 E26 影響：

| 結論 | 出處 |
|---|---|
| 用單一純量定義「匹配失真」，在失真種類不同的兩族之間不成立 | E20 §0 |
| 等 LPIPS 多臂探針：判別一個指標實際在收什麼費 | E20 §5.3、E28 §1 |
| `local_acutance_dev` 對位移不敏感且不可抵銷 | E20 §6 |
| `local_chroma_bias` 分得出連貫色偏與隨機色度雜訊 | E28 §1 |
| ΔE 那一族量的是色度誤差量值，不是人眼在意的軸 | E28 §1.1 |
| SSIM 在保真項中補貼模糊 | E20 §3.3 |
| site S 的鈍化來源是雙線性重取樣 | E20 §5.2 |
| A 族（site L/E/W）的 VAE 重建地板高於加性運作點 | E17–E19 |
| 每一道約束只擋自己那一種失效，且都不循環 | E28 §3 |
| TF32 使同一份程式在不同機器上精度不同 | E28 §4 |

---

## 4. 主網格的設定（已定案）

四道約束的交集，每一個門檻都由實測或人眼定錨：

| 約束 | 值 | 來源 |
|---|---|---|
| `tau_lpips` | 0.02 / 0.05 / 0.10 | 掃描軸 |
| `tau_acut` | 0.04 | E20 §8.1，四個實測值決定 |
| `tau_chroma` | 0.8 | E28 §2，人眼階梯 |
| `beta_linf` | 0（關閉） | E13：L∞ 對非加性不對等 |

其餘：

| 參數 | 值 | 來源 |
|---|---|---|
| `guidance_scale` | 7.5 | E26 §3 |
| `alpha_lpips` | 0 | E27，修訂之四 |
| `margin` | 1.0 | E27 §2 |
| `stop_on_plateau` | 開 | E21–E23 §5.4 |
| `color_max_dev` | 2.0 | E27 §2 |
| site C `lr` / `ranks` | 0.3 / 32 | E27 §3 |
| site P `lr` / `ranks` | 0.03 / 16 | E27 §3 |
| 網格 | 2 site × 3 τ × 6 圖 = 36 格 | site S 已排除（使用者 2026-08-01 決定） |

**判準是語意軸，不是 `net_lpips`。** SigLIP 通過 E25 §1.1 的對照，CLIP 沒有。
`scripts/e27_report.py` 已依此撰寫。

---

## 5. 開跑前必須先做的一件事

`e27d` 那一輪校準是在**還沒有色度約束**時做的。加入第三道之後必須重跑一次
短校準，確認 **`tau_lpips` 仍是綁定者，而不是色度 hinge**。

理由：site C 的解在舊約束下色度偏壓是 4.97，遠超過 0.8，加上約束後一定會被
壓下來；壓下來之後 lr=0.3 還適不適用是未知的。若色度變成綁定者，site C 的
學習率要重新定。

指令（8 格、60 步、約 20 分鐘）：

```bash
for LR in 0.1 0.3; do
  python scripts/run_defense.py --sites C --ranks 32 --lr $LR \
    --size 512 --steps 60 --k_inv 10 --n_edit 10 --limit 2 --no_eval \
    --guidance_scale 7.5 --beta_linf 0 --tau_lpips 0.05 --margin 1.0 \
    --alpha_lpips 0 --color_max_dev 2.0 --out runs/e29_C_lr$LR
done
for LR in 0.03 0.1; do
  python scripts/run_defense.py --sites P --ranks 16 --lr $LR \
    --size 512 --steps 60 --k_inv 10 --n_edit 10 --limit 2 --no_eval \
    --guidance_scale 7.5 --beta_linf 0 --tau_lpips 0.05 --margin 1.0 \
    --alpha_lpips 0 --out runs/e29_P_lr$LR
done
python scripts/e27_binding_check.py runs/e29_*
```

判準：`e27_binding_check.py` 對每一格的判定必須是「LPIPS hinge」，不是硬上界、
不是 margin、不是色度 hinge。

---

## 6. 主網格

校準通過後：

```bash
for TAU in 0.05 0.02 0.10; do
  python scripts/run_defense.py --sites C --ranks 32 --lr <校準值> \
    --tau_lpips $TAU --size 512 --steps 150 --k_inv 10 --n_edit 10 \
    --guidance_scale 7.5 --beta_linf 0 --margin 1.0 --alpha_lpips 0 \
    --color_max_dev 2.0 --stop_on_plateau --out runs/e30_C_tau$TAU
  python scripts/run_defense.py --sites P --ranks 16 --lr <校準值> \
    --tau_lpips $TAU --size 512 --steps 150 --k_inv 10 --n_edit 10 \
    --guidance_scale 7.5 --beta_linf 0 --margin 1.0 --alpha_lpips 0 \
    --stop_on_plateau --out runs/e30_P_tau$TAU
done
python scripts/e27_report.py       # 需把 RUNS 內的名稱改為 e30_*
python scripts/e27_binding_check.py runs/e30_*
```

τ 的順序取 0.05 先跑：那是與 E21/E23 對應的格子，若機時不足至少有關鍵比較。

**成本**（H100 實測 2.5 s/step、40 s/格評測、GPU 使用率 82–92% 故並行無益）：

| 情境 | 平均步數 | 合計 |
|---|---|---|
| 預期 | 60 | 約 2 小時 |
| 上限 150 用滿 | 150 | 約 4.2 小時 |

---

## 7. 不要重走的死路

- `net_lpips` 當防禦成功的判準（E25）
- 對抗性強健的感知度量（E-LPIPS / R-LPIPS / LipSim）——解的是相反的失效
- NLPD、VIF、GMSD、HaarPSI 當保真約束——量的是位移
- ΔE76 / ΔE00 / `dchroma` 當色度約束——量的是量值不是連貫性（E28 §1）
- low rank（使用者 2026-07-30 排除）
- site L / E / W（A 族）——VAE 重建地板高於加性運作點
- site S（使用者 2026-08-01 決定不放進重跑）
- 固定步數的網格

---

## 8. 已知但未修的缺陷

- **`local_dchroma_dev` 已證明無效**，保留為陰性對照，不要拿來當約束。
- **cross-attention 的 `divergence` 模式在 φ=0 梯度為零**。已新增 `suppress`
  模式並改為預設，但 `divergence` 本身未修。
- **`e23_Sbic_s100_tau0.10` 只完成 3/6 圖**，不可用於任何比較。
- **同一個 `max_dev` / `tau_chroma` 對不同影像不是同一個預算**。色度能量跨
  影像差兩倍（E26 §5.4）；本階段採單一固定值，若出現逐圖不一致就要改為逐圖
  正規化。

---

## 9. 環境

- 測試基準 **247 passed / 1 skipped**。
- 本機：`C:/Users/nelso/miniconda3/envs/wacv/python.exe`，torch 2.13.0+cu126，
  RTX 2050 4 GB。**跑得動分析、跑不動主網格**——SD v1.4 光是 fp32 權重就
  4.26 GB，實測 512² 訓練 OOM；256² 可跑但每步 183 s（溢位到系統記憶體），
  且 256² 的結果不可用（所有門檻都在 512² 定的）。
- fp16 目前不受支援（`mat1 and mat2 must have the same dtype`），要用得先把
  整條管線的 dtype 統一並處理數值穩定性。
- **TF32 預設關閉**（`src/utils/device.py`）。`WACV_ALLOW_TF32=1` 可換速度。
- 遠端 Lightning AI Studio：conda env `cloudspace`，背景腳本**不是 login
  shell**，必須用絕對路徑 `/home/zeus/miniconda3/envs/cloudspace/bin/python3`。
  環境重建約 5 分鐘，見 `scripts/drivers/e27_calibration.sh`。
- TWCC 容器已無法登入（金鑰被拒），帳號到期，需由使用者從網頁刪除。

---

## 10. 分支

`claude/e20-fidelity-constraint`，**未併入 main**，已推上 origin。
