# `src/baselines/` 審查報告

> 2026-08-05。審查對象：`src/baselines/{pgd,photoguard,mist,dia,advpaint,promptflare}.py`
> 與 `tests/test_baselines.py`。對照基準：`docs/SOURCE_AUDIT_2026-08-05.md`、
> `docs/_audit_promptflare_photoguard.md`、`docs/_audit_dia_apa.md`、
> `docs/_audit_mist_diffvax.md`。
>
> 審查者未修改任何檔案。所有實測皆以
> `C:/Users/nelso/miniconda3/envs/wacv/python.exe` 執行，未載入 SD 權重
> （以 `diffusers.UNet2DConditionModel` 的小型實例代替）。

---

## 結論摘要

**沒有找到致命等級的缺陷。** 逐項核對的結果是：

| 審查重點 | 結果 |
|---|---|
| 1. 與原始碼的偏離（四項指定檢查） | 四項**全部正確**，見 §0 |
| 2. 值域是否逐篇獨立 | **正確**，五篇各有獨立 `ValueRange` 實例，無共用，見 §0 |
| 3. 梯度是否到得了 `x_adv` | 五篇的可微路徑**皆通**（PromptFlare／AdvPaint 已實測），但 AdvPaint 有一處使 autograd 的保護失效，見 [重要-3] |
| 4. `grad_reps` 平均的位置 | **正確**（先平均梯度再正規化），與 `super_l2` 逐行相同 |
| 5. 投影與夾限的順序 | 五篇**全部正確** |
| 6. 測試是否假通過 | 現有 34 項無假通過，但**覆蓋範圍嚴重不足**，見 [重要-8] |

重要 7 項、次要 6 項，逐項如下。

---

## 0. 已核對且正確的項目（先列出來，避免後續讀成全篇皆有問題）

### 0.1 四項指定檢查全部通過

| 檢查項 | 位置 | 佐證 |
|---|---|---|
| Mist 的損失是 `nn.MSELoss(reduction="sum")`（平方 L2），不是 L2 norm | `mist.py:63` `self.mse_sum = torch.nn.MSELoss(reduction="sum")`，於 `mist.py:162,165` 使用 | `_audit_mist_diffvax.md:79` `self.fn = nn.MSELoss(reduction="sum")` |
| AdvPaint 只用 self-attn 的 Q/K/V + cross-attn 的 **Q** | `advpaint.py:120-124`：`is_self` 時記 `self_q/self_k/self_v`，否則**只**記 `cross_q`；`advpaint.py:264-271` 對四組累加 | `SOURCE_AUDIT §1.3` |
| PromptFlare 記錄 `to_out` 之後的 `A·V·W_out` | `promptflare.py:226-240`：`to_out[0]` → `to_out[1]` → `residual` → `rescale` → 才呼叫 `attn_controller` | `_audit_promptflare_photoguard.md:262-278` |
| PhotoGuard-c 是歸一化梯度 + L2 renorm，不是 sign／L∞ | `pgd.py:291-296`（`normalized_grad`）、`pgd.py:339-341`（`torch.renorm(p=2, dim=0)`）；`photoguard.py:167,171` | `_audit_promptflare_photoguard.md:466-486` |

### 0.2 值域：逐篇獨立，無共用

五個模組各自定義一個 `ValueRange` 實例
（`photoguard.py:53`、`mist.py:44`、`dia.py:53`、`advpaint.py:62`、`promptflare.py:55`），
`DIA_RANGE` 只由同一篇的兩個變體共用，屬正確。

每一處轉換都取 `spec.value_range` 或該模組自己的常數，未見任何一處
硬編碼 `-1/1` 或借用他篇的值域：

- `pgd.py:393,396` `vr = spec.value_range`；`x_ref = vr.from01(x01)`。
- `photoguard.py:142` `vr = ctx.spec.value_range`，`loss_fn` 內做
  `to01 → sdedit → from01`，兩次轉換皆用同一個 `vr`。
- Mist／DIA／AdvPaint／PromptFlare 的 `loss_fn` 直接把 `[-1,1]` 的 `x_adv`
  餵給 `sd.vae.encode`（正確：`sd.encode_image` 會再做一次 `*2-1`，
  五篇都**沒有**誤用它）。

`BaselineSpec.__post_init__`（`pgd.py:159-166`）的 `eps / scale == eps_pixel01`
交叉檢查對 L∞ 與 L2 半徑都成立（兩者皆隨值域線性縮放），數值全部通過。

### 0.3 `grad_reps` 與投影／夾限順序

- `pgd.py:414-425`：`grad_reps` 次前傳 → `torch.stack(grads).mean(0)` →
  `pgd.py:428` 才做 `update_direction`。與
  `_audit_promptflare_photoguard.md:470-478` 的
  `grad = torch.stack(all_grads).mean(0)` 後才 `grad_norm` 逐行相同。
- `pgd.py:428-429`：更新 → `project`；`project`（`pgd.py:335-344`）內部是
  投影 → `clamp(vr.lo, vr.hi)`。與 PhotoGuard（`clamp(X + d_x_norm, ±1)`）、
  AdvPaint、PromptFlare（`min/max` 後 `clamp(-1,1)`）、DIA
  （`eta = clamp(...); clamp(source + eta, ±1)`）、Mist（advertorch
  `batch_clamp` 後 `clamp`）**五篇全部一致**。

### 0.4 已實測可跑的路徑

以小型 `UNet2DConditionModel` 實測（無需 SD 權重）：

- `promptflare.MyAttnProcessor2_0` + `AttnController` 在 diffusers 0.39.0 下
  可安裝、可攔截、`cal_loss` 回傳帶 `grad_fn` 的張量，
  `torch.autograd.grad(loss, z)` 成功（grad norm 0.0476）。
  `Attention.forward` 的 `**cross_attention_kwargs` 在 SD1.x 路徑為空，
  故 `promptflare.py:157-164` 的防禦性 `RuntimeError` 不會誤觸。
- `advpaint.QKVRecorder` 的 forward pre-hook 正確區分 attn1／attn2，
  記到 4/4/4/4 組張量，梯度可回傳。`group_norm` 與 `norm_cross` 實測為 `None`，
  與 `advpaint.py:107-109` 的註解一致。
- `sd.scheduler` 實際是 `PNDMScheduler`（`sd.py:85` 用
  `StableDiffusionPipeline.from_pretrained` 的預設），`dia.py:95` 取的
  `final_alpha_cumprod` 在 PNDM 上存在且等於 `alphas_cumprod[0]`，
  與 DIA 所用 DDIMScheduler（`set_alpha_to_one=False`）同值；`steps_offset`
  兩者皆為 1。**此處不成問題**，但依賴的是兩個排程器碰巧同值。

---

## 重要等級的發現

### [重要] `run_pgd` 的 `seed` 永遠傳不到 `prepare`，三篇 baseline 換 seed 會得到逐位元相同的結果

**位置**：`src/baselines/pgd.py:365`、`src/baselines/pgd.py:400`、`src/baselines/pgd.py:405`

**現況**：

```python
def run_pgd(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    seed: int = 0,            # ← 365，keyword-only，被 run_pgd 自己吃掉
    log_every: int = 20,
    verbose: bool = True,
    **kw,
) -> PGDResult:
    ...
    gen = torch.Generator(device="cpu").manual_seed(seed)      # 400：只給起點用
    x_init_cpu = initial_point(x_ref.cpu(), spec, gen)
    ...
    ctx = spec.prepare(sd, x01, spec, **kw)                    # 405：kw 裡沒有 seed
```

`seed` 是 `run_pgd` 的 keyword-only 參數，因此呼叫端寫 `run_pgd(..., seed=k)`
時 `k` 綁到 `run_pgd` 自己的 `seed`，**永遠不會出現在 `**kw` 裡**，
`spec.prepare` 只會拿到各模組自己的預設值
（`photoguard.py:96` `seed=0`、`mist.py:86` `seed=23`、
`dia.py:152` `seed=1234`、`advpaint.py:201` `seed=9999`、
`promptflare.py:280` `seed=0`）。

各篇的**全部**隨機性（VAE 後驗抽樣、擴散時間步與雜訊、PhotoGuard 每個 rep
的編輯噪聲）都由 `prepare` 建立的 generator 提供，`run_pgd` 的 `seed` 只影響
`initial_point`。而 `photoguard_c`、`promptflare`、`dia_r` 三篇的
`init_rule` 是 `"none"`（`photoguard.py:173`、`promptflare.py:362`、`dia.py:238`），
`initial_point` 直接回傳 `x_ref.clone()`——**這三篇的 `seed` 完全無作用**。

實測（以 stub `sd`／stub `prepare`，`init_rule="none"`）：

```
prepare 收到的 seed = 999   # run_pgd(..., seed=0)
prepare 收到的 seed = 999   # run_pgd(..., seed=12345)
兩次 seed 不同但結果逐位元相同: True
```

**應為**：`prepare` 需要拿到呼叫端指定的 seed。各篇原始碼的隨機性都由一個
全域 seed 控制（Mist `seed_everything(23)`、AdvPaint `AdvPaint.py:317`
`torch.manual_seed(seed)`、DIA `--seed 1234`），並非固定不可變的方法常數；
`_audit_promptflare_photoguard.md:180-186` 亦明記 PromptFlare「全 repo 無 seed，
保護結果不可逐位元重現」，本專案是**主動加上**了 seed 控制
（`promptflare.py:373` 的 `modification_note` 第 (4) 項），那就必須真的可控。

**後果**：一旦實驗協定是「每張圖跑 n 個 seed 取平均與標準差」，
`photoguard_c` / `promptflare` / `dia_r` 三篇的 n 次結果會逐位元相同，
標準差恰為 0；`mist` / `advpaint` / `dia_pt` 只有 PGD 起點不同，
損失端的隨機串流仍完全相同，變異被系統性低估。我方方法若照常用多 seed，
比較會在「變異估計」這一維上不對等。不會有任何錯誤訊息。

**確信度**：確定（已實測重現）。

---

### [重要] AdvPaint 的 GT 與每次迭代使用**不同**的雜訊抽樣，此設計無逐字佐證

**位置**：`src/baselines/advpaint.py:162-190`、`advpaint.py:241-243`

**現況**：

```python
def _forward_and_record(sd, ctx_like, x_paper, emb2, t0, generator, recorder,
                        differentiable: bool):
    ...
        post = sd.vae.encode(x_paper.to(sd.vae.dtype)).latent_dist
        z0 = post.sample(generator) * sd.scaling_factor
        noise = torch.randn(
            z0.shape, generator=generator, device=z0.device, dtype=z0.dtype
        )
```

`prepare` 以同一個 `generator` 先算一次 GT（`advpaint.py:241-243`），
之後每次 `loss_fn` 再抽一次。generator 狀態前進，故 **GT 的 `noise` 與
第 1…100 次迭代的 `noise` 兩兩皆不同**。模組 docstring
（`advpaint.py:168-170`）宣稱這是原作行為：

> `noise` 每次呼叫都是新的一抽——原作對 GT 與每一次迭代都呼叫
> `randn_tensor(..., generator=model.generator)`，產生器狀態會前進，
> 故 GT 與各迭代所用的雜訊本來就不同。此處照樣重現。

**應為**：無法確認。`docs/` 下**沒有 AdvPaint 的逐字佐證檔**——
`_audit_promptflare_photoguard.md`、`_audit_dia_apa.md`、`_audit_mist_diffvax.md`、
`_audit_purify.md` 四份都不含 AdvPaint，唯一的來源是 `SOURCE_AUDIT §1`，
而該節只列了 PGD 更新式、投影、值域、步長衰減、timestep、優化器、mask
七項，**未涉及雜訊抽樣**。`advpaint.py:26-28` 自己也承認
「SOURCE_AUDIT §1.4 未記此項（隨機初始化）」，即該模組有一批細節是
在無留存證據的情況下寫入的。

**後果**：若原作實際上是「GT 與各迭代共用同一個 `latents` 雜訊」，
則現行實作的損失
`Σ‖QKV(GT; ε₁) − QKV(x_adv; ε₂)‖₂` 會被 `ε₁ − ε₂` 造成的差值主導，
梯度方向 `(cur − GT)/‖cur − GT‖` 近乎隨機，AdvPaint 會被系統性削弱，
而輸出仍是一張看起來正常的防禦圖。這是「baseline 被不當弱化」的方向，
正好對我方有利，最需要排除。反之若原作確實每次重抽，現行實作正確。

**確信度**：推測。判定需要 `JoonsungJeon/AdvPaint/AdvPaint.py` 原檔第 200–300 行。

---

### [重要] AdvPaint 的 `total = x_adv.sum() * 0.0` 使 autograd 的「輸入未被使用」保護失效

**位置**：`src/baselines/advpaint.py:263`

**現況**：

```python
    total = x_adv.sum() * 0.0        # 保持 dtype／device，且不影響數值
    for gt_group, cur_group in zip(ctx.gt, cur):
        ...
        for gt_t, cur_t in zip(gt_group, cur_group):
            total = total - (gt_t - cur_t).norm(p=2)
```

**應為**：`pgd.py:421` 用的是
`torch.autograd.grad(loss, probe)`，`allow_unused` 為預設的 `False`。
這正是本專案唯一會偵測「梯度斷了」的機制：若 hook 未安裝、
或 `_forward_and_record` 的路徑被 `no_grad` 切斷，autograd 會拋出
`One of the differentiated Tensors appears to not have been used`。
加上 `x_adv.sum() * 0.0` 這個錨點後，`probe` 恆被使用，該檢查**永遠不會觸發**，
斷掉的路徑只會回傳一個全零梯度。

**後果**：`sign(0) = 0` → `x_adv` 停在初始值（AdvPaint 有 `uniform_linf`
隨機起點，故 delta 不是零而是一張隨機噪點）→ 100 步跑完、log 正常列印、
`loss` 曲線是一條平線但仍有數值。除非有人去看 `delta_linf01` 恆等於 eps
且 loss 完全不變，否則看不出來。這是唯一一篇把安全網拆掉的 baseline
（其餘四篇的損失都直接由 `x_adv` 推導，未被使用時會拋錯）。

現行程式的 hook **確實會觸發**（已實測），所以這不是「已經壞了」，
而是「壞掉時不會有症狀」。

**確信度**：確定（機制），該路徑目前完好亦為確定。

---

### [重要] DIA 的 `prev_timestep = t − 100` 與其自訂格點的間距 111 不一致，且無逐字佐證

**位置**：`src/baselines/dia.py:65-82`、`dia.py:114-127`

**現況**：

```python
def dia_timesteps(sd, num_inference_steps=10):
    ts = np.linspace(0, 1, num_inference_steps) * (n_train - 2)   # 間距 ≈ 110.9
    ...
def _step(sd, ctx, t, z, inversion):
    eps = sd._eps(z, t, ctx.emb)
    prev_t = int(t) - sd.num_train_timesteps // ctx.num_inference_steps   # = t − 100
    a_t = ctx.abar[int(t)]
    a_prev = ctx.abar[prev_t] if prev_t >= 0 else ctx.final_abar
```

格點為 `[1, 112, 223, 334, 445, 555, 666, 777, 888, 999]`（間距 111），
但每一步的 α 對是 `(abar[t], abar[t-100])`——**步進 100 與格點間距 111 不相符**，
軌跡的 α 不相接。

**應為**：`_audit_dia_apa.md:274-276` 只記到

> **timestep 排程**：自訂 `custom_set_timesteps`（`attack/DIA_PT.py:165-192`），
> `timesteps = round(linspace(0,1,10) * 998) + 1e-6`，再加 `steps_offset`；反演時不反轉順序。

`custom_diffusion` 的本體（`dia.py:118` 引用的 `DIA_PT.py:300-322`）
與 `backward_ddim`（引用 `utils/utils_general_H.py:32-34`）
**在四份 `_audit_*.md` 中都查不到原文**。另外 `dia.py:70-72` 稱
「原始碼以 `inversion_flag=False` 呼叫（降冪），再於反演迴圈中 `reversed(...)` 走訪」，
與 audit 的「反演時不反轉順序」敘述相反（兩者的淨結果都是升冪，
故不影響數值，但顯示兩份說法至少有一份不精確）。

**後果**：若 DIA 原始碼的 `prev_timestep` 取的是**相鄰格點**（t − 111）
而非 t − 100，則整條 10 步反演軌跡的 α 排程不同，DIA-PT 的損失
（反演終點對 `E(x_adv)` 的偏離）與 DIA-R 的重建誤差都會落在不同量級上，
兩個變體的強度會系統性偏移。輸出仍是一張正常的防禦圖。

**確信度**：推測。判定需要 `sohn1029/DIA` 的 `attack/DIA_PT.py:300-322` 原文。

---

### [重要] PhotoGuard-c 的 VAE 編碼取 `mean`，原作取 `sample()`，且此偏離未列入 `modification_note`

**位置**：`src/baselines/photoguard.py:144-154` →（間接）`src/models/sd.py:145-160`

**現況**：`loss_fn` 走 `sd.sdedit(...)`，`sdedit`（`sd.py:470`）呼叫
`self.encode_image(x01, use_ckpt=vae_ckpt)`，而 `encode_image`（`sd.py:156,160`）
固定取 `latent_dist.mean`：

```python
        return self.vae.encode(x).latent_dist.mean * self.scaling_factor
```

**應為**：`_audit_promptflare_photoguard.md:392-393`（`attack_forward` 原文）：

```python
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
```

即原作取**後驗抽樣**。Mist 與 DIA 兩篇都為此另寫了取樣路徑
（`mist.py:66-74` 的 `_encode_sampled`、`dia.py:130-136` 的直通式），
並在各自的 docstring 說明理由；PhotoGuard-c **沒有**做同樣處理，
`photoguard.py:178-186` 的六條 `modification_note` 也未列此項。

**後果**：兩個層面。一是忠實度：`grad_reps=10` 在原作中是為了對
「`eta=1` 的 DDIM 隨機性 + 未固定的 `latents` + VAE 後驗抽樣」三重隨機性取期望，
本移植只保留了編輯噪聲一項，期望值的對象與原作不同。二是紀錄完整性：
`modified_from_paper=True` 的條目若漏列偏離，報表會把該行讀成已完整揭露，
而 `docs/CLAUDE.md` 與 `tests/test_baselines.py:1-9` 的整個設計前提正是
「漏標比不標更糟」。

**確信度**：確定（`sd.py:156,160` 與 audit 原文可直接對照）。

---

### [重要] AdvPaint、DIA、PromptFlare 有多處引用的行號在 `docs/` 中查不到佐證

**位置**：`advpaint.py:1-54`（全篇）、`dia.py:33-44`、`dia.py:107-127`、
`promptflare.py:308`

**現況**：以下宣稱在四份 `_audit_*.md` + `SOURCE_AUDIT` 中**都找不到對應原文**：

| 宣稱 | 引用 | 佐證狀態 |
|---|---|---|
| AdvPaint 有均勻隨機初始化 | `AdvPaint.py:80` | 無（`advpaint.py:26-28` 自承 SOURCE_AUDIT 未記） |
| AdvPaint 的可微路徑只走 masked-image | `AdvPaint.py:203-212` | 無 |
| AdvPaint 的 CFG 兩列嵌入、`--prompt` required | `AdvPaint.py:97, 317, 346` | 無 |
| DIA-PT「值取樣、梯度走均值」 | `DIA_PT.py:355, 388` + `vjp_encode_fn` | `_audit_dia_apa.md:112` 只記到 `.sample()`，未提 `.mode()` |
| DIA-R 前向與反傳都用 `.mode()` | `DIA_R.py:353` | 無 |
| `backward_ddim` 的展開式 | `utils_general_H.py:32-34` | 無 |
| PromptFlare 的兩列嵌入是**同一個 prompt** | `promptflare.py:22` | 無 |

其中 `promptflare.py:308` 的 `emb2 = emb.repeat(2, 1, 1)` 影響最大：
若原作那兩列其實是 `[uncond; cond]`（CFG 形式），則
`AttnController` 的 `pred`（第 0 列）與 `target`（第 1 列）就同時差了
「BOS 遮罩」與「prompt 內容」兩個變因，損失的語意完全不同。
從方法邏輯（BOS decoy 需要單一變因）推斷同 prompt 是合理的，但**無佐證**。

**後果**：這些是「別人論文的方法」的關鍵細節。查證檔的存在理由就是讓後續
任何人能重新核對；沒有留存原文時，一旦實驗跑完出現異常結果，
無從判定是方法問題還是移植問題，而 TWCC 容器已刪、實驗無法重跑。

**確信度**：確定（已對四份 audit 檔逐項 grep）。

---

### [重要] 測試完全沒有覆蓋 `prepare` / `loss_fn` / `run_pgd`，`step_size` 與 `objective` 也不在對照表內

**位置**：`tests/test_baselines.py` 全檔

**現況**：34 項測試（實跑：`34 passed, 1 xfailed`）全部只驗三類東西：

1. `BaselineSpec` 的**欄位值**（`AUDIT` 對照表，`test_baselines.py:29-36`）；
2. `BaselineSpec.__post_init__` 的建構期驗證；
3. `initial_point` / `project` / `update_direction` / `step_size_at` 四個純函式。

**五篇的 `loss_fn` 與 `prepare` 一行都沒有被執行過**，`run_pgd` 的骨幹迴圈
也沒有。`_minimal()`（`test_baselines.py:119`）用的
`loss_fn=lambda *a, **k: torch.zeros(())` 只服務於建構期測試，未進入任何
數值路徑。

`AUDIT` 對照表（`test_baselines.py:29-36`）只釘 5 欄
（`eps`、`eps_pixel01`、`norm`、`steps`、`update_rule`）。**未被任何測試釘住的欄位**：

- `step_size`（與 eps 同等重要，且是三篇有值域陷阱的參數之一：
  Mist 2/255、PromptFlare 2/255、DIA 1/255、AdvPaint 0.03、PhotoGuard 1.0）
- `objective`（只有 `photoguard_c` 與 `dia_pt` 兩篇被斷言，
  `test_baselines.py:316-317`；mist／advpaint／promptflare／dia_r 未釘）
- `init_rule`（只有 mist／advpaint 被間接斷言）
- `grad_reps`（只有 photoguard_c 被斷言）
- `step_schedule`（advpaint 有；其餘由 `test_固定步長的四篇不衰減` 間接涵蓋，
  但該測試在 `step_schedule == "constant"` 時是恆真的——
  `step_size_at` 的 constant 分支直接回傳 `spec.step_size`，
  兩次呼叫必然相等。它實際驗的只是「沒被設成 linear_decay」）

**後果**：`step_size` 被改成一個「看起來整齊」的值（例如把 PromptFlare 的
2/255 寫成 `[0,1]` 尺度的 2/255）不會有任何測試失敗，而那正是
`SOURCE_AUDIT §10` 整節在防的錯誤。`objective` 被寫反會讓攻擊變成保護，
同樣無測試攔截（`test_baselines.py:313-317` 的 docstring 自己寫了
「方向弄反會讓攻擊變成保護」，卻只斷言兩篇）。

**確信度**：確定。

---

## 次要等級的發現

### [次要] Mist 的 `z_target` 被快取，原作每次前傳都重新抽樣

**位置**：`src/baselines/mist.py:111-112`

**現況**：

```python
    with torch.no_grad():
        z_target = _encode_sampled(sd, tgt_paper, generator).detach()
```

**應為**：`_audit_mist_diffvax.md:77-84` 顯示 `zy` 是在 `forward` **內部**取得的：

```python
        zx, loss_semantic = self.get_components(x, True)
        zy, _ = self.get_components(self.target_info, True)
```

`get_components` 內是 `get_first_stage_encoding(...)` → `.sample()`，
故原作每次前傳的 textural target 都帶一次新的 VAE 後驗抽樣噪聲。

**後果**：textural 是 16,384 個元素的平方和，target 的後驗噪聲對總損失只貢獻
一個小的常數偏移，對梯度方向的影響可忽略。屬忠實度紀錄問題，
不預期改變結論。

**確信度**：確定（偏離存在），影響量級為推測。

---

### [次要] PromptFlare 的 attention processor 在例外時不會還原

**位置**：`src/baselines/promptflare.py:265-269`、`src/baselines/pgd.py:447-448`

**現況**：`PromptFlareContext.__init__` 在 `prepare` 階段就換掉全部 attn2 的
processor，而還原只發生在 `run_pgd` 迴圈**正常結束後**：

```python
    if hasattr(ctx, "close"):
        ctx.close()
```

400 步 × 512² × 全 activation 保留的 OOM 風險不低；一旦中途拋出例外，
processor 留在 UNet 上。

**後果**：後續共用同一個 `SDWrapper` 的實驗會走進 `MyAttnProcessor2_0`。
不過該 processor 的數學與 `AttnProcessor2_0` 等價，且 batch=1 時
`hidden_states.chunk(2)` 只會回傳一個張量，`pred, target = ...` 立即拋
`ValueError`——**是響亮的失敗而非靜默污染**，故列為次要。
`try/finally` 或 context manager 可解決。

**確信度**：確定。

---

### [次要] PhotoGuard-c 在空 prompt 下仍走兩次 UNet 前向，成本加倍而數值不變

**位置**：`src/baselines/photoguard.py:126-127, 150-153`

`emb = sd.encode_text("")`、`emb_uncond = sd.encode_text("")`，兩者相同；
`_eps_cfg`（`sd.py:246-248`）算 `eps_u + 7.5*(eps_c − eps_u)`，
因 `eps_c == eps_u` 而恆等於 `eps_u`。原作亦然
（`text_embeddings = cat([uncond, text])` 且 `prompt=""`），故**忠實**，
但 200 步 × 10 reps × 4 去噪步的攻擊成本因此整整多一倍。
`_eps_cfg` 已有 `guidance_scale == 1.0` 的短路分支可用。

**確信度**：確定。

---

### [次要] `docs/RUNBOOK_2026-08-05.md:167` 與程式不一致

**現況**：RUNBOOK 寫「AdvPaint 250」，`advpaint.py:282` 與
`test_baselines.py:83-84` 已依 `SOURCE_AUDIT §1.1` 的裁決取 100。
另 `docs/LOGIC_CHECK_2026-08-05.md:37` 引用的
`test_baseline_registry.py` 不存在（實際檔名為 `tests/test_baselines.py`）。

**確信度**：確定。

---

### [次要] `SOURCE_AUDIT §11` 的 DIA-PT 起點裁決仍未定，程式維持 (a) 照抄不改

**位置**：`src/baselines/pgd.py:180-237`、`tests/test_baselines.py:259-288`

`_l1_projection` 的 `if c2.nelement != 0`（`pgd.py:211,226`——比較的是方法物件，
恆為真）以及 `[-1,1]` 值域下投影分支被跳過的缺陷，**確實是 AutoAttack／DIA
原始碼自身的問題**，本專案的轉寫無誤，且已由 `xfail(strict=True)` 釘住。
此處僅提醒：`SOURCE_AUDIT §11` 建議的 (b)「起點加一次 `project()`」尚未執行，
在此之前 DIA-PT 的失真預算不受控，「同 τ_LPIPS 比較」對它不成立。

**確信度**：確定（現況描述）。

---

### [次要] `photoguard.py` 對 `sdedit` 輸出被 `clamp(0,1)` 一事已記錄，但該記錄只在 docstring

**位置**：`src/baselines/photoguard.py:39-42`

`decode_latent`（`sd.py:183`）回傳 `((x+1)/2).clamp(0,1)`，原作
`attack_forward` 回傳未裁切的 `vae.decode(...).sample`；被裁切的像素梯度為零。
此事寫在模組 docstring，也寫進了 `modification_note` 第 (6) 項——
**已正確處置**，此處僅記錄它與 DIA-R 的作法不一致：`dia.py:189` 為了同一個理由
繞過了 `decode_latent`，直接呼叫 `sd.vae.decode`。兩篇對同一個問題採不同對策。

**確信度**：確定。

---

## 我無法確認的項目

以下各項需要 GPU、SD 權重、或原始 repo 檔案才能判定，本次審查未做：

1. **AdvPaint 原始碼**（`JoonsungJeon/AdvPaint/AdvPaint.py`）。需確認：
   GT 與各迭代是否共用同一個 `latents` 雜訊（[重要-2]）；
   第 80 行的隨機初始化；第 203–212 行的 `no_grad` 範圍；
   第 97/317/346 行的 `guidance_scale` / `seed` / `preprocess`。
   `docs/` 下無任何 AdvPaint 逐字佐證檔。
2. **DIA 原始碼**（`sohn1029/DIA`）。需確認：
   `custom_diffusion`（`DIA_PT.py:300-322`）的 `prev_timestep` 定義（[重要-4]）；
   `backward_ddim`（`utils_general_H.py:32-34`）的展開式；
   `vjp_encode_fn` 是否真的用 `.mode()`；DIA-R 的 `_encode_r`；
   `tmp_latent` 是否與軌跡起點共用同一次 `.sample()`。
3. **PromptFlare `promptflare.py:22`** 的 `prompt_embeds` 建構方式
   （兩列是同一個 prompt，還是 `[uncond; cond]`）。
4. **實跑一次 `run_pgd`**：五篇的 `prepare` + `loss_fn` 從未被執行過。
   最低限度應在真實 SD v1.4 上跑 2–3 步，確認
   (a) 損失是純量且有限、(b) `torch.autograd.grad` 回傳非零梯度、
   (c) `delta_linf01` 在預算內、(d) 512² 下的 peak VRAM
   （PhotoGuard-c 200 步 × 10 reps × 4 步去噪、PromptFlare 400 步全 activation
   保留，兩者都可能 OOM）。
5. **PromptFlare 的 `loss_depth` 白名單在真實 SD v1.4 UNet 上是否命中**。
   `prepare`（`promptflare.py:292-299`）已強制 512×512，理論上 attn2 的
   token 數為 `{4096, 1024, 256, 64}`，白名單命中後三者；但未在真實
   UNet 上實測（本次只在小型 UNet 上驗證機制可跑）。
   若一個都沒命中，`cal_loss` 會回傳 Python `int 0`，
   `pgd.py:417` 的 `loss.dim()` 會拋 `AttributeError`——會有症狀，但值得先確認。
6. **`torch.renorm(d_x, p=2, dim=0, maxnorm=16)` 在 `[1,3,512,512]` 上的實際幾何**。
   `_audit_promptflare_photoguard.md` 附錄 B.3 已實測，本次未複驗。
7. **Mist 的 `MIST.png`** 是否已取得。`mist.py:99-104` 在缺 target 時拋
   `NotImplementedError`，故 Mist 目前**跑不起來**；需確認該檔已入庫。
8. **`src/baselines` 尚未被任何腳本呼叫**。全 repo grep `run_pgd` 只命中
   `src/baselines/` 自身與 `tests/`，`scripts/run_defense.py` 未整合。
   因此上述所有問題都還沒有污染任何已存在的 run。

---

## 附：本次審查實際執行的指令

```
python -m pytest tests/test_baselines.py -q          # 34 passed, 1 xfailed
python - <<  (小型 UNet2DConditionModel 上驗證 MyAttnProcessor2_0 / AttnController)
python - <<  (同上，驗證 QKVRecorder 的 attn1/attn2 區分與梯度)
python - <<  (stub sd + stub prepare，重現 run_pgd 的 seed 不傳遞)
python - <<  (fp16 CPU 上驗證 initial_point 的三種規則)
```
