# 目標函數：主流影像免疫法的兩個形狀

派工 `scripts/immunise_field.py`，設定 `configs/immunise_field_objective.json`。
載體固定在 8 px 預算，**自變數只有目標函數**。

## 兩個主流項

實作在 `src/defense/mainstream_terms.py`，都不含文字、不含任何編輯指令。

**`enc_target` — PhotoGuard 的 targeted encoder attack。** 最小化
`‖E(x_def) − z_target‖ / ‖z_target‖`，把 latent 拉到一個固定的目標上。
目標取**中性灰**而不是另一張真實照片：真實照片會把一個具體身分寫進目標，
而那張照片的選擇會變成一個沒有理由的自由度。

**`diffusion` — AdvDM／Mist 的擴散訓練損失。** 最大化
`E_t ‖ε_θ(z_t, t, c_img) − ε‖²`，`z_t` 由**防禦圖自己的 latent** 加噪而成。
前兩項問「模型的預測偏掉多少」，這一項問「模型有多不會還原這張圖」。
加噪走該時刻自己的 σ 而不是 `scheduler.add_noise`，理由與
`instruction_free._scaled_sample` 相同（排程器是有狀態的）。

## 量到什麼

| 變體 | 目標 | 過半穿透 |
|---|---|---|
| `obj_longchain` | 基準，但取樣鏈 6→12 步、反傳 1→2 | 7/25 |
| `obj_diffusion` | `cond` 換成 `diffusion` | 6/25 |
| `obj_targeted` | `enc` 換成 `enc_target` | 5/25 |
| `obj_baseline` | 現行的 `enc`／`cond`／`id` | 1/25 |
| `obj_all` | 五項並存，權重不調 | 1/25 |

## 這張表讀不出排名

`obj_baseline` 與 `runs/field_ladder/` 的 `flow_08` 是**同一個設定**。
那裡量到 **5/25**，這裡量到 **1/25**——同設定重跑的差就是 1 對 5，
與表上的全部差距同量級。代理項也一樣：`flow_08` 的代理 id 中位是 0.695，
`obj_baseline` 是 0.533，差 0.16，而五個變體彼此只差 0.533–0.681。

所以**三個主流目標在這個樣本數下沒有超出重跑雜訊地勝過基準**。
門檻掃描也不一致：門檻 0.7 時 `obj_baseline` 反而是 16、最高的之一。
詳見 `docs/EVALUATION.md`。

`obj_all` 的 1/25 是那一格的設計目的——問五項並存會不會互相抵消。它沒有相乘。

## 要分辨得出來，得先修可重現性

要嘛設決定性旗標並接受變慢，要嘛同一設定重跑數次、報分布而不是報單點。
在那之前，再加變體也分辨不出哪一個目標函數比較好。
