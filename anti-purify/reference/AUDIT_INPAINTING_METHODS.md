# AdvPaint／DIA／PromptFlare 原始碼逐字佐證

> 2026-08-05。本檔補上 `src/baselines/` 審查（已刪，見 `INDEX.md` §3）點名的證據缺口：
> `src/baselines/{advpaint,dia,promptflare}.py` 有多項宣稱在既有四份
> `_audit_*.md` 與 `SOURCE_AUDIT` 中查不到原文。以下全部取自官方 repo 的
> raw 檔，逐字貼出，並標明該段落回答了哪一個待決問題。
>
> 取得方式與檔案指紋（同一批次下載，皆 HTTP 200）：
>
> | 檔案 | 來源路徑 | bytes |
> |---|---|---|
> | `AdvPaint.py` | `JoonsungJeon/AdvPaint` `main` | 13891 |
> | `DIA_PT.py` | `sohn1029/DIA` `attack/` | 17763 |
> | `DIA_R.py` | `sohn1029/DIA` `attack/` | 18872 |
> | `promptflare.py` | `NAHOHYUN-SKKU/PromptFlare` | 4261 |
> | `attention_control.py` | `NAHOHYUN-SKKU/PromptFlare` | 5187 |
> | `protect.py` | `NAHOHYUN-SKKU/PromptFlare` | 2281 |
>
> 本檔所有行號皆以上表的 raw 檔為準。既有模組 docstring 中若干行號取自
> 另一版本，已在本次一併更正（見 §4）。

---

## 判定總表

| 待決問題（來自 `src/baselines/` 的程式審查） | 判定 | 依據 |
|---|---|---|
| [重要-2] AdvPaint 的 GT 與各迭代是否共用同一次雜訊抽樣 | **不共用，本專案實作正確** | §1.3 |
| [重要-4] DIA 的 `prev_timestep` 是 t−100 還是相鄰格點 t−111 | **t−100，且此不一致是原作自身的，本專案實作正確** | §2.2 |
| PromptFlare 的兩列嵌入是同一 prompt 還是 `[uncond; cond]` | **同一 prompt，靠 `encoder_attention_mask` 區分** | §3.1 |
| AdvPaint 有均勻隨機初始化（`AdvPaint.py:80`） | **有，實際行號 76** | §1.1 |
| AdvPaint 可微路徑只走 masked-image（`AdvPaint.py:203-212`） | **是，實際行號 206-214** | §1.2 |
| AdvPaint 的 CFG 兩列嵌入、`--prompt` required（`:97,317,346`） | **是，實際行號 100／318／335** | §1.4 |
| DIA-PT「值取樣、梯度走均值」（`DIA_PT.py:355,388`） | **是，實際行號 255（`.mode()`）與 343/357（`.sample()`）** | §2.3 |
| DIA-R 前向與反傳都用 `.mode()`（`DIA_R.py:353`） | **是，行號 342／353 皆為 `.mode()`** | §2.4 |
| `backward_ddim` 的展開式（`utils_general_H.py:32-34`） | **與本專案 `_ddim` 逐字相同，實際位於 `DIA_PT.py:19-20`** | §2.1 |
| `tmp_latent` 是否與軌跡起點共用同一次 `.sample()` | **是，同一個 `xx`** | §2.5 |

---

## 1. AdvPaint（`AdvPaint.py`）

### 1.1 隨機初始化（回答「`AdvPaint.py:80`」）

實際位於第 76 行，且在 mask 迴圈（第 82 行 `for mask_num, m in enumerate(mask_dir)`）
**之外**：

```python
72:    X_ori = X.detach().clone()
74:    mask_dir = glob.glob(args.mask_dir+"/*.png")
76:    X_adv = X.clone().detach() + ((torch.rand(*X.shape)*2*eps-eps).to("cuda"))
82:    for mask_num, m in enumerate(mask_dir):
```

即：uniform ℓ∞ 起點確實存在（`SOURCE_AUDIT §1.4` 未記此項），且多個 mask
**共用同一個 `X_adv`**、逐 mask 累積 100 步。本專案的 img2img 版把 mask
設為全圖、只跑一次，該累積不適用。

### 1.2 可微路徑（回答「`AdvPaint.py:203-212`」）

實際位於 206-214。plain image 那一路整段被 `torch.no_grad()` 包住：

```python
200:            X_adv.requires_grad_(True)
205:            ## Plain image ##
206:            with torch.no_grad():
207:                image_latents = model.vae.encode(X_adv).latent_dist.sample(model.generator)
208:                image_latents = model.vae.config.scaling_factor * image_latents
213:                noise = randn_tensor(image_latents.shape, generator=model.generator, ...)
214:                latents = model.scheduler.add_noise(image_latents, noise, model.latent_timestep)

218:            ## Masked image ##
219:            masked_image = X_adv * (mask_512 < 0.5)
222:            masked_image_latents = model.vae.encode(masked_image).latent_dist.sample(model.generator)
```

故梯度只經第 219-226 行的 masked-image 分支（9 通道輸入的後 4 通道）。
img2img 沒有該通道，本專案改由加噪後的 latent 施力——這是**實質**改動，
已列於模組 docstring 的移植表第 2 列。

### 1.3 GT 與各迭代的雜訊（回答 [重要-2]）

GT 段：

```python
123:        with torch.no_grad():
124:            image_latents = model.vae.encode(X_ori).latent_dist.sample(model.generator)
128:            noise = randn_tensor(image_latents.shape, generator=model.generator, device=model.device, dtype=prompt_embeds.dtype)
129:            zT = model.scheduler.add_noise(image_latents, noise, model.latent_timestep)
```

迭代段（第 207-214 行，見 §1.2）呼叫的是**同一個** `model.generator`。
產生器狀態在每次 `.sample()` 與 `randn_tensor` 之後前進，故：

- GT 的 `noise` ≠ 第 1 次迭代的 `noise` ≠ 第 2 次…… 兩兩皆不同。
- 即使 `model.generator` 為 `None`（第 100 行的 pipeline 呼叫未傳 generator），
  `randn_tensor` 退回全域 RNG，而全域 RNG 同樣在每次抽樣後前進
  （第 326 行 `torch.manual_seed(seed)` 只在程式起點設一次），結論不變。

**判定**：`src/baselines/advpaint.py` 的 `_forward_and_record` 共用同一個
`ctx.generator`、每次呼叫重抽，與原作一致。[重要-2] 所擔心的
「baseline 被系統性削弱」**不成立**。

### 1.4 CFG、seed、eps（回答「`:97,317,346`」）

```python
100:            model(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=7.5)
318:    seed = 9999
335:        X = preprocess(init_image).half().to("cuda")
338:        adv_X = pgd_SelfQKV_And_Cross_Xadv(img_dir, X, model=pipeline,
344:                    clamp_min=-1, clamp_max=1)
370:    parser.add_argument('--prompt', required=True, help='prompt')
372:    parser.add_argument('--eps', default=0.1, type=float)
373:    parser.add_argument('--step_size', default=0.05, type=float)
375:    parser.add_argument('--iters', default=100, type=int)
```

對照函式簽章（第 57 行）：

```python
57: def pgd_SelfQKV_And_Cross_Xadv(img_dir, X, model, eps=0.06, step_size=0.03, iters=100, clamp_min=0, clamp_max=1, mask_num=1):
```

三點確認：`guidance_scale=7.5` 使 `do_classifier_free_guidance` 為真、
`prompt_embeds` 為兩列；`--prompt` required 無預設；值域由 `main` 傳入的
`clamp_min=-1, clamp_max=1` 決定（簽章預設的 `0/1` 未被使用）。
eps 的**兩組預設互相矛盾**（簽章 0.06／CLI 0.1），`SOURCE_AUDIT §1.1`
取 0.06／0.03／100 步，本專案照此。

### 1.5 步長衰減與更新式

```python
197:            actual_step_size = step_size - (step_size - step_size / 100) / iters * iter
287:            loss = (loss_query + loss_key + loss_value + loss_cross_q) / length
290:            grad, = torch.autograd.grad(loss, [X_adv])
295:            X_adv = X_adv - grad.detach().sign() * actual_step_size
296:            X_adv = torch.minimum(torch.maximum(X_adv, X_ori - eps), X_ori + eps)
297:            X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
```

線性衰減至 `step_size/100`；更新 → 投影 → 夾限；`objective="minimize"`。
四項損失皆為 `-=` 累加後除以 `length`（第 262 行
`length = len(self_query[timestep].keys())`，即單一 timestep 下的 path 數）。
`length` 為常數，sign 更新下不改變方向。

---

## 2. DIA（`DIA_PT.py`／`DIA_R.py`）

### 2.1 `backward_ddim`（回答「`utils_general_H.py:32-34` 查不到」）

該函式實際就定義在 `DIA_PT.py` 檔頭：

```python
19: def backward_ddim(x_t, alpha_t: "alpha_t", alpha_tm1: "alpha_{t-1}", eps_xt):
20:     return ((alpha_tm1**0.5*alpha_t**-0.5)*x_t + alpha_tm1**0.5*((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt)
```

與 `src/baselines/dia.py` 的 `_ddim` 逐字相同。

### 2.2 `prev_timestep` 與格點間距（回答 [重要-4]）

格點：

```python
182:    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
184:    timesteps = np.linspace(0, 1, num_inference_steps) * (self.config.num_train_timesteps-2) # T=999
185:    timesteps = timesteps + 1e-6
186:    timesteps = timesteps.round().astype(np.int64)
188:    if not inversion_flag:
189:        timesteps = np.flip(timesteps).copy()
191:    self.timesteps = torch.from_numpy(timesteps).to(device)
192:    self.timesteps += self.config.steps_offset
```

`linspace(0,1,10) * 998` 的間距為 110.9，四捨五入後 `+steps_offset=1`
得 `[1, 112, 223, 334, 445, 555, 666, 777, 888, 999]`，相鄰間距 111。

DDIM 步：

```python
299:            prev_timestep = (tt[0]-self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps)
310:            prev_timestep = (tt-self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps)
315:        alpha_prod_t_prev = (
316:            self.model.scheduler.alphas_cumprod[prev_timestep]
317:            if prev_timestep >= 0
318:            else self.model.scheduler.final_alpha_cumprod
319:        )
320:        if inversion:
321:            alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
322:        res_latent=backward_ddim(z, alpha_prod_t, alpha_prod_t_prev,eps_xt=cfg_noise)
```

`1000 // 10 = 100`，故 `prev_timestep = t − 100`。

**判定**：「步進 100 與格點間距 111 不相符」是**原作自身**的性質，不是本專案
的移植錯誤。`src/baselines/dia.py::_step` 的 `prev_t = int(t) - sd.num_train_timesteps //
ctx.num_inference_steps` 與之逐字相同，`final_alpha_cumprod` 的退路與
反演時交換 α 的順序亦相同。[重要-4] 所擔心的「兩個變體強度系統性偏移」
**不成立**（若要「修正」成 t−111，反而會偏離原作）。

註：第 188-189 行在 `inversion_flag=False`（預設）時降冪排列，
而反演迴圈以 `reversed(timestep_tensor)`（第 267、347、360 行）走訪，
淨結果為升冪。`_audit_dia_apa.md:274-276` 記為「反演時不反轉順序」，
與此不符；以本檔為準，`dia.py:70-72` 的敘述正確。

### 2.3 DIA-PT 的值／梯度雙路徑（回答「`DIA_PT.py:355,388`」）

```python
255:        vjp_encode_fn = lambda x: (self.model.vae.encode(x.contiguous().reshape(img_shape)).latent_dist.mode()*0.18215).view(-1)
343:            xx = self.model.vae.encode(x_adv).latent_dist.sample()*0.18215      # get_clean_X_T
357:            xx = self.model.vae.encode(x_adv).latent_dist.sample()*0.18215      # get_grad
377:                vae_out,vjp=torch.autograd.functional.vjp(vjp_encode_fn,inputs.view(-1), v=lam.detach().clone().reshape(-1))
```

前向的**值**由 `.sample()` 取得，反傳的 vjp 走 `.mode()`。
確認 `src/baselines/dia.py::_encode_pt` 的作法正確。

### 2.4 DIA-R 全程用 `.mode()`（回答「`DIA_R.py:353`」）

```python
342:            xx = self.model.vae.encode(x_adv).latent_dist.mode()*0.18215        # get_clean_X_T
353:            xx = self.model.vae.encode(x_adv).latent_dist.mode()*0.18215        # get_grad
257:        vjp_encode_fn = lambda x: (... .latent_dist.mode()*0.18215).view(-1)
258:        vjp_decode_fn = lambda x: (self.model.vae.decode(((1 / 0.18215)*x).contiguous().reshape(latent_shape)).sample).view(-1)
378:            loss=(self.model.decode_image(inputs).float() - x_adv.detach().float()).norm(p=2)
```

兩個變體的差別因此有三處，不只損失定義：DIA-R 的前向取 `.mode()`
（DIA-PT 取 `.sample()`），且損失是**像素域**的重建誤差
`‖D(x_T) − x_adv‖₂`，比較對象是當前的 `x_adv` 而非原圖。

### 2.5 `tmp_latent` 與軌跡起點（回答審查未決項）

```python
357:            xx = self.model.vae.encode(x_adv).latent_dist.sample()*0.18215
358:            tmp_latent = xx.clone()
360:            for tt in reversed(timestep_tensor):
361:                mid_x.append(xx.detach().clone())
362:                xx = self.custom_diffusion(tt, xx, ...)
368:        loss=(inputs-tmp_latent.detach()).norm(p=2).float()
```

`tmp_latent` 就是軌跡起點 `xx` 的複本，**同一次 `.sample()`**。
故損失是「反演終點對其自身起點的偏離」，不含額外的抽樣變異。

### 2.6 更新式與 L1 起點

```python
102: def grad_normalize(source_latent,X_adv,grad,i,eps=20,step_size=10,clamp_min=-200,clamp_max=200,iters=100):
103:     adv_image=X_adv+step_size*grad.sign()
104:     eta=torch.clamp(adv_image-source_latent,-eps,eps)
105:     X_adv=torch.clamp(source_latent+eta,clamp_min,clamp_max).detach()

241:        if self.norm == 'L1':
242:            t = torch.randn_like(source_latents).to(device).detach()
243:            delta = L1_projection(source_latents, t, self.eps)
244:            x_adv = source_latents + t + delta
248:        x_adv = x_adv.clamp(-1., 1.).to(dtype=source_latents.dtype)
```

更新為 sign **上升**（`objective="maximize"`），順序仍是更新 → 投影 → 夾限。
L1 起點的缺陷（`L1_projection` 第 54 行 `if c2.nelement != 0:` 比較的是
方法物件、恆為真；且該函式的 `u = torch.min(1 - x - y, x + y)` 假設值域為
`[0,1]` 而 DIA 用 `[-1,1]`）確認為**原始碼自身**的問題，本專案的轉寫無誤，
已由 `tests/test_baselines.py` 的 `xfail(strict=True)` 釘住。

---

## 3. PromptFlare（`promptflare.py`／`attention_control.py`／`protect.py`）

### 3.1 兩列嵌入是同一個 prompt（回答「`promptflare.py:22`」）

```python
13:    text_inputs = pipe.tokenizer(prompt, padding="max_length",
16:        max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
20:    text_input_ids = text_inputs.input_ids
21:    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]
22:    text_embeddings = text_embeddings.repeat(2, 1, 1) # [2, 77, 768]
23:    text_embeddings = text_embeddings.detach()
```

`repeat(2,1,1)` 作用在**單一** prompt 的嵌入上，兩列完全相同。
`[uncond; cond]` 的疑慮**排除**。兩列的差異全部來自注意力遮罩：

```python
40:    encoder_attention_mask = torch.ones(2, 77).to(device=pipe.device)
41:    encoder_attention_mask[1][1:] = 0
50:        noise_pred = pipe.unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings, encoder_attention_mask=encoder_attention_mask)[0]
51:        pred_noise, target_noise = noise_pred.chunk(2)
```

第 1 列只保留位置 0（BOS），即 decoy 目標；第 0 列不遮罩。
單一變因成立，方法邏輯與本專案 `promptflare.prepare` 的實作一致。

### 3.2 `loss_depth` 白名單與步數

```python
8:    num_inference_steps = 4
9:    k = 1
10:    loss_mask = True
11:    loss_depth = [1024, 256, 64]
29:    noisy_model_input_shape = (1, num_channels_latents, 64, 64)
44:    for i in range(min(k, num_inference_steps)):
53:        text_loss = attn_controller.cal_loss(loss_mask=loss_mask, loss_depth=loss_depth)
58:    loss = torch.stack(text_losses).mean()
```

`k=1` 使迴圈只跑一次，`torch.stack(...).mean()` 對單一元素取平均、
第 56 行 `latents = pred_noise` 是死碼——確認 `src/baselines/promptflare.py`
的 `loss_fn` docstring 所述正確。

`loss_depth = [1024, 256, 64]` 是 **token 數**白名單，對應 latent 64²
（512² 影像）下 attn2 的 32²／16²／8² 三層，**排除**最外層的 4096。
1024² 影像的 latent 為 128²，各層 token 數為 `{16384, 4096, 1024, 256}`，
與白名單的交集只剩 `{1024, 256}` 兩層且對應的相對深度不同，
故 `prepare` 對非 512² 直接拋 `NotImplementedError` 是正確處置——
沿用原值會涵蓋錯的層且無症狀。

### 3.3 更新式

```python
75:    grad_reps = 1
94:            grad = torch.autograd.grad(loss, [cur_masked_adv])[0] * (1 - cur_mask)
95:            grad = grad.detach().sum(dim=0, keepdim=True)
102:        avg_grad = torch.stack(grads).mean(0)
104:        adv = adv - avg_grad.detach().sign() * args.step_size
105:        adv = torch.minimum(torch.maximum(adv, src_image_orig - args.eps), src_image_orig + args.eps)
106:        adv.data = torch.clamp(adv, min=-1.0, max=1.0)
```

`grad_reps=1`、sign 下降、更新 → 投影 → 夾限、值域 `[-1,1]`。
另確認第 73 行的攻擊 prompt 是固定的 quality-tag 字串
（`"professional photography, best quality, ..."`），非使用者輸入；
本專案為 prompt-free，該字串的處置見 `SOURCE_AUDIT §2`。

---

## 4. 因本次查證而更正的行號

`src/baselines/advpaint.py` 的模組 docstring 原引用的行號取自另一版本，
已更正為本檔表列 raw 檔的行號：

| 項目 | 原引用 | 更正為 |
|---|---|---|
| 隨機初始化 | `AdvPaint.py:80` | `AdvPaint.py:76` |
| `no_grad` 範圍 | `AdvPaint.py:203-212` | `AdvPaint.py:206-214` |
| `guidance_scale=7.5` | `AdvPaint.py:97` | `AdvPaint.py:100` |
| `torch.manual_seed(seed)` | `AdvPaint.py:317` | `AdvPaint.py:318`（`seed = 9999`），`manual_seed` 於 326 |
| `preprocess` | `AdvPaint.py:346` | `AdvPaint.py:335` |

`dia.py` 與 `promptflare.py` 的引用行號經核對無誤，未更動。

---

## 5. 仍未取得的原始檔

- AdvPaint 的 `pipeline_stable_diffusion_inpaint_pgd.py`、`utils_UNet.py`
  （`self_query` 等全域字典的實際記錄時機與層清單）。本專案的
  `QKVRecorder` 以 forward pre-hook 自行實作，形狀與分組已在小型 UNet 上實測。
- PromptFlare 的 `attention_control.py` 已下載但本檔尚未逐行貼出
  （`cal_loss` 的跨層加總已由 `_audit_promptflare_photoguard.md` 佐證）。
- Mist 的 `MIST.png` 目標影像仍未取得，`data/targets/` 只有 `gray.png`；
  在取得之前 `mist.prepare` 會拋 `NotImplementedError`，Mist 跑不起來。
