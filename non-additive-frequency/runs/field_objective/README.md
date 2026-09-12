# 目標函數：把主流影像免疫法的兩個形狀接進來

派工 `scripts/immunise_field.py`，設定 `configs/immunise_field_objective.json`，
鏈式腳本 `scripts/objective_chain.sh`。載體固定在 `flow_08` 的預算
（臉框 8 px、背景仿射殘差 1 px、摺疊 0.001），**自變數只有目標函數**。

## 現行目標的形狀

`src/defense/instruction_free.py` 的 `enc` 與 `cond` 都是**無目標**的相對位移：

    enc  = ‖E(x_def) − E(x)‖ / ‖E(x)‖
    cond = mean_t ‖ε(·, c_img(x_def)) − ε(·, c_img(x))‖ / ‖ε_orig‖

兩個都只要求「離原來的地方遠」，不指定要去哪裡。梯度會隨著離開原點而轉向，
所以同樣的預算可能繞路。文獻上被採用的兩個作法都不是這個形狀。

## 兩個主流項

實作在 `src/defense/mainstream_terms.py`，兩個都不含文字、不含任何編輯指令。

**`enc_target` — PhotoGuard 的 targeted encoder attack。**
最小化 `‖E(x_def) − z_target‖ / ‖z_target‖`，把 latent 拉到一個固定的目標上。
整條路徑指向同一個點，不繞路。目標取**中性灰**而不是另一張真實照片：
真實照片會把一個具體身分寫進目標，而那張照片的選擇會變成一個沒有理由的自由度。

**`diffusion` — AdvDM／Mist 的擴散訓練損失。**
最大化 `E_t ‖ε_θ(z_t, t, c_img) − ε‖²`，`z_t` 由**防禦圖自己的 latent** 加噪
而成。前兩項問「模型的預測偏掉多少」，這一項問「模型有多不會還原這張圖」——
被推向的是訓練分布之外，而不只是離原圖遠。IP2P 的 UNet 吃九個通道，
`z_t` 與影像條件兩側都用防禦圖：攻擊者拿到的就是這張圖，兩側本來就一致。

加噪不走 `scheduler.add_noise`：`EulerAncestralDiscreteScheduler` 是有狀態的
（`step_index` 只在第一次呼叫時依 timestep 初始化），逐時刻的獨立探針連續呼叫
會全部套用第一個時刻的 σ。改成用該時刻自己的 σ 做無狀態加噪
`z_t = (z_0 + σ_t·ε) / √(σ_t²+1)`，與 `instruction_free._scaled_sample`
同一個理由、同一個式子。

## 五個變體

| 變體 | 權重 | 問什麼 |
|---|---|---|
| `obj_baseline` | id 1.0、enc 0.5、cond 1.0 | 與 `runs/field_ladder/` 的 `flow_08` 逐字相同，是這一批自己的對照 |
| `obj_targeted` | enc → `enc_target` | targeted 的梯度形狀值不值錢 |
| `obj_diffusion` | cond → `diffusion` | 推出訓練分布與推遠預測，哪一個有效 |
| `obj_all` | 五項並存 | 合起來會不會互相抵消 |
| `obj_longchain` | 同基準，鏈 6→12、反傳 1→2 | 代理與真編輯的落差值不值得付兩倍時間 |

`obj_targeted` 用 `enc_target` **取代** `enc` 而不是並存：兩者量的是同一個
VAE latent，並存會對同一個量重複計權。`obj_diffusion` 與 `cond` 同理。
`obj_all` 不調權重——那一格的用途是看會不會抵消，不是找最佳權重。

## 為什麼要量鏈長

`runs/field_ladder/` 的 `flow_16` 代理 `id` 項是 0.32，而真編輯上的
`id_norm` 中位是 0.584。代理**高估**了防禦。代理跑 6 步取樣鏈、只對末端一步
反傳，真編輯跑 50 步；`obj_longchain` 把鏈加長到 12 步、反傳 2 步，量那個落差
收斂多少、成本翻幾倍。

## 符號慣例

`CombinedObjective.score` 一律回傳**要最小化**的那個方向：

    score = w_id·id − w_enc·tanh(enc) − w_cond·tanh(cond)
            + w_enc_target·tanh(enc_target) − w_diffusion·tanh(diffusion)

`enc_target` 越小越好所以是加號，`diffusion` 越大越好所以是減號。
權重為零的項**不計算**——每一項都要跑一輪 UNet，關掉的項不該付那個成本，
`tests/test_mainstream_terms.py` 的 `test_權重為零的項不計算` 釘住這一條。
