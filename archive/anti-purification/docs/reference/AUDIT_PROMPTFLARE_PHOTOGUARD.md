# Baseline 原始碼查證：PromptFlare 與 PhotoGuard

查證日期：2026-08-05。所有引用均取自各 repo `main` 分支的 raw 檔案。

| Repo | HEAD SHA | HEAD 日期 |
|---|---|---|
| `NAHOHYUN-SKKU/PromptFlare` | `5e9ad004b0e06cfe69608a65e38d2aa24f870142` | 2025-12-16 |
| `MadryLab/photoguard` | `686bea75c786cb46c88fc396a0cd0ee3d7d28c2e` | 2023-02-27 |

以下所有行號皆為上述 HEAD 版本之檔案行號。凡未在原始碼中出現者，一律列入
文末「未找到的項目」，不作推測。

---

# PromptFlare

## 1. `cal_loss` 完整計算式

### 1.1 原始碼原文（`attention_control.py` 第 37–54 行）

```python
37	    def cal_loss(self, loss_mask, loss_depth):
38	        text_losses = 0
39	
40	        attn2_preds = self.attn2_preds
41	        for i, (pred, target) in enumerate(zip(attn2_preds, self.attn2_targets)):
42	            _, h, _ = pred.shape
43	
44	            text_loss = 0
45	            if h in loss_depth:
46	                if loss_mask:
47	                    pred = pred * self.masks[h]
48	                    target = target * self.masks[h]
49	                text_loss = (pred - target.detach()).norm(p=2)
50	            text_losses += text_loss
51	
52	        self.reset()
53	
54	        return text_losses
```

`pred` / `target` 的來源在同檔第 11–20 行：

```python
11	    def __call__(self, hidden_states, module_name):
12	        hidden_states = hidden_states.clone()
13	
14	        pred, target = hidden_states.chunk(2) # e.g. [1, 256, 1280]
15	        _, h, _ = pred.shape
16	
17	        self.make_mask(h)
18	        if module_name.endswith('attn2'):
19	            self.attn2_preds.append(pred)
20	            self.attn2_targets.append(target)
```

mask 的降採樣在第 26–32 行：

```python
26	    def make_mask(self, h):
27	        if h not in self.masks:
28	            size = int(np.sqrt(h))
29	
30	            new_mask = torch.nn.functional.interpolate(self.masks[512], size=(size, size))
31	            new_mask = new_mask.flatten().unsqueeze(0).unsqueeze(2)
32	            self.masks[h] = new_mask
```

### 1.2 計算式

設 `attn2` 層集合為 $\mathcal{L}$，第 $l$ 層輸出的 token 數為 $h_l$（= 該層 latent
空間解析度的平方），$O_l^{(0)}$ 為 batch 第 0 項（無 BOS mask）之 attention 模組
輸出，$O_l^{(1)}$ 為 batch 第 1 項（僅 BOS 可被 attend）之輸出，$M_{h_l}$ 為
降採樣後的影像遮罩，則

$$
L = \sum_{l \in \mathcal{L},\; h_l \in \texttt{loss\_depth}}
\bigl\lVert\, M_{h_l} \odot O_l^{(0)} \;-\; \mathrm{sg}\bigl[M_{h_l} \odot O_l^{(1)}\bigr] \,\bigr\rVert_2
$$

其中 $\lVert\cdot\rVert_2$ 是對整個張量取 Frobenius norm（`torch.Tensor.norm(p=2)`
的預設行為，非逐列 L2、非平方），$\mathrm{sg}[\cdot]$ 為 `detach()`。

三點須注意：

1. **是「加總」不是「平均」。** 第 50 行 `text_losses += text_loss`。論文 Eq. 12
   寫的是 $\arg\min \mathbb{E}_l[L^l_{CA}]$（對層取期望）。由於外層更新用
   `sign()`（見 §2.4），常數倍率不影響更新方向，但若要重現數值化的 loss 曲線
   必須知道這個差異。
2. **被 detach 的是 BOS-masked 分支**（`target`），即最佳化把「正常 cross-attention
   輸出」拉向「只 attend BOS 的輸出」。論文 Eq. 11 未指明哪一側 detach。
3. **不是平方**。Eq. 11 寫 $\lVert\cdot\rVert_2$，程式碼亦為 `.norm(p=2)`，一致。

### 1.3 `loss_mask` 與 `loss_depth`

| 參數 | 型別 | 意義 | 預設值 | 出處 |
|---|---|---|---|---|
| `loss_mask` | `bool` | 為 `True` 時，第 47–48 行把 `pred`、`target` 逐元素乘上 `self.masks[h]`，使損失只在遮罩區累計 | `True` | `promptflare.py:10` `loss_mask = True` |
| `loss_depth` | `list[int]` | **不是「深度層數」，而是允許計入損失的 token 數 $h$ 的白名單**。第 45 行 `if h in loss_depth`，$h$ 為該 attn2 層的 sequence length | `[1024, 256, 64]` | `promptflare.py:11` `loss_depth = [1024, 256, 64]` |

`cal_loss` 本身沒有預設參數（第 37 行為位置引數），唯一的呼叫點是
`promptflare.py:53`：

```python
53	        text_loss = attn_controller.cal_loss(loss_mask=loss_mask, loss_depth=loss_depth)
```

`loss_depth` 的三個值由 `make_mask` 第 28 行 `size = int(np.sqrt(h))` 可知對應
latent 解析度 32×32、16×16、8×8。SD v1 inpainting UNet 另有 $h = 4096$
（64×64）的 attn2 層，該層不在白名單內、損失為 0。此與論文
「$l$ is set to include only the attention layers excluding the outermost one」
的敘述一致；惟本次查證未載入模型實測各 attn2 層的實際 $h$ 值，該對應關係
係由 `int(np.sqrt(h))` 之算術推得。

### 1.4 遮罩極性（實測驗證）

`protect.py:40` 對 mask 圖做 `ImageOps.invert`，其後：

- `promptflare.py:82` `masked_adv = adv * (1 - cur_mask)`
- `promptflare.py:94` `grad = torch.autograd.grad(...)[0] * (1 - cur_mask)`
- `cal_loss` 用 `self.masks[h]`（即 `cur_mask` 本身）遮罩損失

以 repo 內 `sample/` 實測（原圖 vs inpaint 輸出、原圖 vs protected 輸出）：

- 原始 mask 檔白色區佔 15.5%，`invert` 後白色區佔 84.5%。
- inpainting 只改動 `invert` 後為白（=1）的區域（該區 88.9% 的像素改變，
  原始白區 0% 改變）。→ **`invert` 後 `cur_mask == 1` 即待重繪區**。
- 對抗噪聲只加在 `1 - cur_mask`（即原始 mask 白區、會被保留的區域），
  該區平均像素差 4.05、最大 6.33；另一區平均 0.31。

結論：損失在待重繪區計算，噪聲加在保留區。與論文 Eq. 11「comparison is made
only over the region indicated by the image latent mask $M'$」一致。

### 1.5 BOS mask 的實作路徑

```python
40	    encoder_attention_mask = torch.ones(2, 77).to(device=pipe.device)
41	    encoder_attention_mask[1][1:] = 0
```
（`promptflare.py:40–41`）

第 0 列全 1（不遮罩），第 1 列只有索引 0（BOS）為 1。此張量經
`pipe.unet(..., encoder_attention_mask=...)` 傳入。已核對 diffusers v0.21.2
（repo `requirements.txt` 釘選版本）：

- `models/unet_2d_condition.py:806–809`
  ```python
  # convert encoder_attention_mask to a bias the same way we do for attention_mask
  if encoder_attention_mask is not None:
      encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
      encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
  ```
  即論文 Eq. 9 的常數 $c = -10000.0$。
- `models/attention.py:218–222`：`BasicTransformerBlock` 把
  `encoder_attention_mask` 當作 `attn2` 的 `attention_mask` 引數傳入，因此會
  進到 `MyAttnProcessor2_0.__call__` 的 `attention_mask` 分支（第 92–96 行）。

## 2. `protect.py` 預設值

### 2.1 argparse 全部項目（`protect.py:47–55`）

| 參數 | argparse flag | dest | 預設值 | 型別 | 出處行號 |
|---|---|---|---|---|---|
| 影像資料夾 | `--image` | `image_folder_path` | `./sample/original_images` | str | `protect.py:48` |
| 遮罩資料夾 | `--mask` | `mask_folder_path` | `./sample/masks` | str | `protect.py:49` |
| 輸出資料夾 | `--output` | `output_folder_path` | `./sample/protected_images` | str | `protect.py:50` |
| 噪聲預算 | `--eps` | `eps` | **12**（`int`） | int | `protect.py:52` |
| 步長 | `--step` | `step_size` | **2**（`float`） | float | `protect.py:53` |
| 迭代次數 | `--epochs` | `epochs` | **400** | int | `protect.py:54` |

**`protect.py` 沒有 `--seed`，全 repo 亦無 `torch.manual_seed` / `np.random.seed`
/ `Generator`。** 已用 `grep -rn -i "seed\|manual_seed\|Generator"` 對全部 5 個
`.py` 檔確認（repo 根目錄只有 `attention_control.py`、`inpaint.py`、
`promptflare.py`、`protect.py`、`utils.py`）。`promptflare.py:30`
`latents = torch.randn(...)` 因此每次執行都不同，**PromptFlare 的保護結果不可
逐位元重現**。

### 2.2 eps / step 的實際數值

```python
23	    args.eps /= 255.0
24	    args.step_size /= 255.0
```
（`protect.py:23–24`）

因此傳入最佳化迴圈的是 `eps = 12/255 ≈ 0.0471`、`step_size = 2/255 ≈ 0.00784`。
**但這兩個數是施加在 $[-1,1]$ 值域的張量上**（見 §5），所以在標準
$[0,255]$ 像素尺度下的實際 L∞ 預算是 $12/255 \times 127.5 = 6.0$ 灰階。

實測佐證：repo 內 `sample/protected_images/a=orange_o=cat_s=real.png` 與
`sample/original_images/`（依 `utils.prepare_image` 的 BICUBIC 縮放至 512）相減，
**最大絕對差為 7，直方圖絕大多數落在 ≤ 6**（7 只有 1090 個像素，屬 fp16 與
量化捨入）。若 eps 真為 12/255 的像素預算，最大差應接近 12。

### 2.3 硬編碼於 `promptflare.py` 的最佳化超參數（無法由 CLI 調整）

| 參數 | 值 | 出處 |
|---|---|---|
| `num_inference_steps`（排程器步數） | 4 | `promptflare.py:8` |
| `k`（實際反傳的時間步數） | 1 | `promptflare.py:9` |
| `loss_mask` | `True` | `promptflare.py:10` |
| `loss_depth` | `[1024, 256, 64]` | `promptflare.py:11` |
| `grad_reps`（梯度平均次數） | 1 | `promptflare.py:75` |
| latent 形狀 | `(1, 4, 64, 64)` | `promptflare.py:29` |
| VAE scaling factor | `0.18215` | `promptflare.py:37` |
| quality tag prompt | 見下 | `promptflare.py:73` |

```python
73	    quality_tag_prompt = "professional photography, best quality, ultra high res, photo, art, high quality, realistic, anime, masterpiece, best quality, artistic, detail, 4k, 8k"
```

因 `k = 1`，第 44 行迴圈只跑一次，第 56 行 `latents = pred_noise` 為不會被使用
到的死碼。實際只在 `timesteps_all[0]`（即 4 步排程的最大 t）上計算一次損失。

### 2.4 PGD 更新（`promptflare.py:104–106`）

```python
104	        adv = adv - avg_grad.detach().sign() * args.step_size
105	        adv = torch.minimum(torch.maximum(adv, src_image_orig - args.eps), src_image_orig + args.eps)
106	        adv.data = torch.clamp(adv, min=-1.0, max=1.0)
```

標準 L∞ sign-PGD，**無隨機初始化**（`promptflare.py:71` `adv = src_image_orig.clone()`），
**無步長衰減**，clamp 到 $[-1, 1]$。

### 2.5 評估階段（`inpaint.py`，非最佳化參數但影響比較基準）

| 參數 | flag | 預設值 | 出處 |
|---|---|---|---|
| 模型 | `--model` | `runwayml/stable-diffusion-inpainting` | `inpaint.py:40` |
| CFG scale | `--guidance_scale` | 7.5 | `inpaint.py:41` |
| 推論步數 | `--inference` | 50 | `inpaint.py:42` |
| strength | `--strength` | 1.0 | `inpaint.py:43` |
| batch size | `--batch` | 10 | `inpaint.py:50` |

`inpaint.py` 同樣**沒有 seed**。

## 3. attention processor 攔截內容

`MyAttnProcessor2_0`（`attention_control.py:56–143`）攔截的是 **cross-attention
模組的最終輸出張量**，形狀 `[batch, tokens, inner_dim]`。

關鍵在第 122–142 行：

```python
122	        hidden_states = F.scaled_dot_product_attention(
123	            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
124	        )
125	        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
126	        hidden_states = hidden_states.to(query.dtype)
127	
128	        # linear proj
129	        hidden_states = attn.to_out[0](hidden_states, scale=scale)
130	        # dropout
131	        hidden_states = attn.to_out[1](hidden_states)
...
136	        if attn.residual_connection:
137	            hidden_states = hidden_states + residual
138	
139	        hidden_states = hidden_states / attn.rescale_output_factor
140	
141	        if self.module_name.endswith('attn2'): # or self.module_name.endswith('attn1'):
142	            self.attn_controller(hidden_states, self.module_name)
143	        return hidden_states
```

逐項回答：

- **不記錄 Q / K / V。** `query`、`key`、`value`（第 101、108–109 行）只作為
  `F.scaled_dot_product_attention` 的輸入，沒有任何一個被傳給 `attn_controller`。
- **不記錄 softmax 後的機率。** 使用 PyTorch 2.0 的
  `F.scaled_dot_product_attention`（融合核），attention map 從未被實體化，
  程式碼中不存在該張量。論文 Eq. 9 的 $\mathcal{A}$ 只是概念層次的描述。
- **記錄的是 $\mathcal{A}V$ 再經 `to_out` 之後的結果**，不是論文 Eq. 10 的
  $\mathrm{CA} = \mathcal{A}V$。已核對 diffusers v0.21.2
  `models/attention_processor.py:73`：`residual_connection: bool = False`、
  第 72 行 `rescale_output_factor: float = 1.0` 為 `Attention` 的預設值，故
  SD UNet 的 attn2 實際上不加 residual、不除 factor；差異僅在
  `to_out[0]`（線性投影 $W_{out}$）與 `to_out[1]`（dropout，推論時為 identity）。
  即記錄值為 $\mathcal{A}V W_{out}$。
- **哪些層：** `promptflare.py:66–68` 對 UNet 中**所有**名稱以 `attn2` 結尾的
  模組安裝此 processor（涵蓋 down / mid / up blocks）：
  ```python
  66	    for n, m in pipe.unet.named_modules():
  67	        if n.endswith('attn2'): # or n.endswith('attn1'):
  68	            m.set_processor(MyAttnProcessor2_0(attn_controller, n))
  ```
  `attn1`（self-attention）被註解掉，不攔截。層的篩選發生在 `cal_loss`
  的 `loss_depth` 白名單，而非在 processor 安裝階段。

## 4. 模型 id

```python
11	    pipe = StableDiffusionInpaintPipeline.from_pretrained(
12	        "runwayml/stable-diffusion-inpainting",
13	        variant="fp16",
14	        torch_dtype=torch.float16,
15	    ).to("cuda")
```
（`protect.py:11–15`）

- 保護階段：`runwayml/stable-diffusion-inpainting`，fp16 variant，`torch.float16`。
- 評估階段預設同上（`inpaint.py:40`），另加 `safety_checker=None`、
  `use_safetensors=True`（`inpaint.py:15–16`）。

環境（`requirements.txt`）：`diffusers==0.21.2`、`transformers==4.27.3`、
`huggingface_hub==0.25.0`；README 指定 `torch==2.4.1` + cu124、Python 3.10。

## 5. 值域

**$[-1, 1]$。**

```python
 6	def prepare_image(image):
 7	    image = image.resize((512, 512), resample=Image.BICUBIC)
 8	    image = np.array(image)
 9	    image = image.transpose(2, 0, 1)
10	    image = torch.from_numpy(image).to(dtype=torch.half) / 127.5 - 1.0
11	
12	    return image
```
（`utils.py:6–12`）

配合 `promptflare.py:106` 的 `torch.clamp(adv, min=-1.0, max=1.0)`，整條最佳化
路徑都在 $[-1,1]$。輸出時 `tensor_to_pil`（`utils.py:25–32`）以
`if tensor.min() < 0: tensor = (tensor + 1) / 2` 轉回 $[0,1]$。

遮罩則為 $[0,1]$，且 `utils.py:19–20` 做硬性二值化：
`mask[mask != 1] = 0; mask[mask == 1] = 1`（BICUBIC 縮放後只有精確等於 1.0 的
像素保留為 1）。

**對重現的直接影響：** 由於 eps 在 $[-1,1]$ 上施加，`--eps 12` 的實際像素域
L∞ 預算是 **6/255**，不是 12/255。若本專案要在「匹配人眼可辨失真」下與
PromptFlare 比較，必須以 6/255 為基準，或把 PromptFlare 的 `--eps` 設為 24
才能得到 12/255 的像素預算。

## PromptFlare 未找到的項目

1. **`--seed` 或任何隨機種子設定**：全 repo 不存在。
2. **loss 中除了 cross-attention 項以外的任何項**（如 LPIPS、L2 影像項、
   正則項）：不存在，`cal_loss` 的回傳值即完整目標。
3. **`E_l`（對層取期望）的實作**：程式碼是加總（`attention_control.py:50`），
   未找到取平均的程式碼。
4. **論文 Eq. 10 的 $\mathcal{A}V$（未經 `to_out`）之擷取點**：不存在。
5. **多時間步的損失聚合**：`k = 1` 硬編碼，`promptflare.py:58`
   `torch.stack(text_losses).mean()` 實際上只對單一元素取平均。
6. **EditBench 資料集的下載或前處理腳本**：repo 內僅有 3 張 sample 影像，
   未找到論文所用 240 張 EditBench 的取得流程。
7. **評估指標（CLIP Score / LPIPS / SSIM / PSNR）的計算程式碼**：repo 內不存在。
8. **`--eps` 的 `type=int`**：`protect.py:52` 為 `type=int`，因此無法從 CLI
   指定非整數的 eps（例如 `--eps 12.5` 會直接報錯）。這不是「未找到」，
   而是需注意的限制。

---

# PhotoGuard

repo 檔案清單（`git/trees/main?recursive=1` 實查）：`notebooks/` 下有
`demo_complex_attack_inpainting.ipynb`、`demo_simple_attack_img2img.ipynb`、
`demo_simple_attack_inpainting.ipynb`、`generating_fake_images.ipynb`、
`utils.py`；另有 `demo/app.py`、`demo/utils.py`。**沒有** `demo_complex_attack_img2img.ipynb`。

## 1. complex attack（diffusion attack）完整損失式與 PGD 迴圈

以下全部出自 `notebooks/demo_complex_attack_inpainting.ipynb`，cell 8（code）。

### 1.1 可微分前向

```python
def attack_forward(
        self,
        prompt: Union[str, List[str]],
        masked_image: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
    ):
        ...
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = text_embeddings.detach()

        num_channels_latents = self.vae.config.latent_channels
        latents_shape = (1 , num_channels_latents, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)

        mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
        mask = torch.cat([mask] * 2)

        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
        masked_image_latents = torch.cat([masked_image_latents] * 2)

        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        for i, t in enumerate(timesteps_tensor):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        return image
```

注意：**沒有 `torch.no_grad()`，整條 `num_inference_steps` 步的去噪迴圈都在
計算圖中**，這正是論文所述需要 A100 40GB 的原因。

### 1.2 損失與梯度

```python
def compute_grad(cur_mask, cur_masked_image, prompt, target_image, **kwargs):
    torch.set_grad_enabled(True)
    cur_mask = cur_mask.clone()
    cur_masked_image = cur_masked_image.clone()
    cur_mask.requires_grad = False
    cur_masked_image.requires_grad_()
    image_nat = attack_forward(pipe_inpaint,mask=cur_mask,
                               masked_image=cur_masked_image,
                               prompt=prompt,
                               **kwargs)

    loss = (image_nat - target_image).norm(p=2)
    grad = torch.autograd.grad(loss, [cur_masked_image])[0] * (1 - cur_mask)

    return grad, loss.item(), image_nat.data.cpu()
```

損失式即

$$
L = \bigl\lVert f(x_{adv}) - x_{targ} \bigr\rVert_2
$$

其中 $f$ 是完整 SD inpainting pipeline 的 VAE 解碼輸出（像素域，$[-1,1]$），
$\lVert\cdot\rVert_2$ 為整張量的 Frobenius norm（**不是平方**；論文 Eq. 5 寫的是
$\lVert\cdot\rVert_2^2$）。梯度乘 `(1 - cur_mask)`：只更新非重繪區。

### 1.3 PGD 迴圈（實際使用的 L2 版本）

```python
def super_l2(cur_mask, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps = 5, target_image = 0, **kwargs):
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    for i in iterator:

        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(cur_mask, X_adv, prompt, target_image=target_image, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)

        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        l = len(X.shape) - 1
        grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        grad_normalized = grad.detach() / (grad_norm + 1e-10)

        # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        actual_step_size = step_size
        X_adv = X_adv - grad_normalized * actual_step_size

        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)

    torch.cuda.empty_cache()

    return X_adv, last_image
```

同一 cell 亦定義了 L∞ 版本 `super_linf`（更新為
`X_adv = X_adv - grad.detach().sign() * actual_step_size`，投影為
`torch.minimum(torch.maximum(X_adv, X - eps), X + eps)`），但在 notebook 中
**未被執行**（見 §2）。

兩個版本的步長衰減式都被註解掉，實際 `actual_step_size = step_size`（定值）。

## 2. eps、step_size、iterations、grad_reps 的實際值

### 2.1 complex attack（cell 10，**實際執行**的 cell）

```python
prompt = ""
SEED = 786349
torch.manual_seed(SEED)

strength = 0.7
guidance_scale = 7.5
num_inference_steps = 4

cur_mask, cur_masked_image = prepare_mask_and_masked_image(init_image, mask_image)

cur_mask = cur_mask.half().cuda()
cur_masked_image = cur_masked_image.half().cuda()
target_image_tensor = prepare_image(target_image)
target_image_tensor = 0*target_image_tensor.cuda() # we can either attack towards a target image or simply the zero tensor

result, last_image= super_l2(cur_mask, cur_masked_image,
                  prompt=prompt,
                  target_image=target_image_tensor,
                  eps=16,
                  step_size=1,
                  iters=200,
                  clamp_min = -1,
                  clamp_max = 1,
                  eta=1,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=guidance_scale,
                  grad_reps=10
                 )
```

| 參數 | 值 | 備註 |
|---|---|---|
| norm | **L2**（`super_l2`） | 非 L∞ |
| `eps` | **16** | `torch.renorm(d_x, p=2, dim=0, maxnorm=16)`，是整張 512×512×3 影像在 $[-1,1]$ 值域下的 L2 半徑，**不是** 16/255 |
| `step_size` | **1** | 定值，無衰減 |
| `iters` | **200** | |
| `grad_reps` | **10** | 每次迭代重跑 10 次前向並平均梯度 |
| `clamp_min/max` | −1 / 1 | |
| `num_inference_steps`（攻擊內部） | **4** | 非 50，論文所稱「backpropagate through only a few steps」 |
| `guidance_scale`（攻擊內部） | 7.5 | |
| `eta` | 1 | DDIM 隨機性；與未固定的 `latents` 共同構成 `grad_reps` 平均的理由 |
| `prompt`（攻擊時） | `""` 空字串 | |
| `SEED` | 786349 | 全域 `torch.manual_seed`，僅在迴圈外呼叫一次 |

`strength = 0.7` 在 cell 10 被賦值但**未傳入** `super_l2`，對攻擊無作用。

### 2.2 complex attack 的 L∞ 版本（cell 11，**整段為註解，未執行**）

```python
## Alternatively you can run an l_inf pgd attack
# result, last_image= super_linf(cur_mask, cur_masked_image,
#                   prompt=prompt,
#                   target_image=target_image_tensor,
#                   eps=0.1,
#                   step_size=0.006,
#                   iters=200,
#                   clamp_min = -1,
#                   clamp_max = 1,
#                   height = 512,
#                   width = 512,
#                   eta=1,
#                   num_inference_steps=num_inference_steps,
#                   guidance_scale=guidance_scale,
#                  )
```

`eps=0.1`、`step_size=0.006`、`iters=200`（無 `grad_reps`，取 `super_linf`
的預設值 5）。因作用於 $[-1,1]$，換算像素域為 L∞ ≈ **12.75/255**、步長
≈ 0.765/255。

### 2.3 simple attack（encoder attack）三處實作

| 出處 | eps | step_size | iters | 值域 | 目標 |
|---|---|---|---|---|---|
| `demo_simple_attack_img2img.ipynb` cell 9 | 0.06 | 0.02 | 1000 | $[-1,1]$ | 零 latent（loss 無 target 項） |
| `demo_simple_attack_inpainting.ipynb` cell 11 | 0.06 | 0.01 | 1000 | $[-1,1]$ | 目標圖 latent |
| `demo/app.py:71–73` | 0.12 | 0.01 | 200 | $[-1,1]$ | 灰階均勻圖 latent |

換算為像素域 L∞：0.06 → **7.65/255**；0.12 → **15.3/255**。

simple attack 的 PGD 有兩個 complex attack 沒有的性質
（`demo_simple_attack_img2img.ipynb` cell 7）：

```python
def pgd(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  

        X_adv.requires_grad_(True)

        loss = (model(X_adv).latent_dist.mean).norm()
        ...
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    

        if mask is not None:
            X_adv.data *= mask

    return X_adv
```

1. **隨機初始化**：`X_adv = X + U(-eps, eps)`。
2. **線性步長衰減**：從 `step_size` 降到 `step_size/100`。

inpainting 版本的簽名多了 `target` 與 `criterion`；`criterion=torch.nn.MSELoss()`
被傳入但**從未使用**，實際損失為
`loss = (model(X_adv).latent_dist.mean - target).norm()`（Frobenius norm，非 MSE）。

## 3. target image 如何取得與使用

### 3.1 complex attack

```python
target_url = "https://i.pinimg.com/originals/18/37/aa/1837aa6f2c357badf0f588916f3980bd.png"
response = requests.get(target_url)
target_image = Image.open(BytesIO(response.content)).convert("RGB")
target_image = target_image.resize((512, 512))
```
（cell 6；另有兩個被註解掉的 `target_url`）

但 cell 10 隨即把它歸零：

```python
target_image_tensor = prepare_image(target_image)
target_image_tensor = 0*target_image_tensor.cuda() # we can either attack towards a target image or simply the zero tensor
```

**因此實際執行的 complex attack，其 target 是全零張量**，在 $[-1,1]$ 值域中
對應 RGB (127.5, 127.5, 127.5) 的純灰影像。下載的圖片不參與計算。損失退化為
$\lVert f(x_{adv}) \rVert_2$，即把 pipeline 的輸出壓成中性灰。

複製此 baseline 時不需要外部網址（該 pinimg 連結亦可能已失效）。

### 3.2 simple attack

- img2img 版：**沒有 target**，`loss = (model(X_adv).latent_dist.mean).norm()`，
  等價於 target = 零 latent。
- inpainting 版（cell 9、11）：從 Boston Globe CDN 下載一張圖，
  `target = pipe_inpaint.vae.encode(preprocess(target_image).half().cuda()).latent_dist.mean`。
- `demo/app.py:50–55`：target 為
  `https://www.rtings.com/images/test-materials/2015/204_Gray_Uniformity.png`
  （灰階均勻測試圖），與論文正文「gray image」一致。

## 4. 值域

**$[-1, 1]$。** `notebooks/utils.py`：

```python
19	def preprocess(image):
20	    w, h = image.size
21	    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
22	    image = image.resize((w, h), resample=Image.LANCZOS)
23	    image = np.array(image).astype(np.float32) / 255.0
24	    image = image[None].transpose(0, 3, 1, 2)
25	    image = torch.from_numpy(image)
26	    return 2.0 * image - 1.0
...
28	def prepare_mask_and_masked_image(image, mask):
29	    image = np.array(image.convert("RGB"))
30	    image = image[None].transpose(0, 3, 1, 2)
31	    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
32	
33	    mask = np.array(mask.convert("L"))
34	    mask = mask.astype(np.float32) / 255.0
35	    mask = mask[None, None]
36	    mask[mask < 0.5] = 0
37	    mask[mask >= 0.5] = 1
38	    mask = torch.from_numpy(mask)
39	
40	    masked_image = image * (mask < 0.5)
41	
42	    return mask, masked_image
```

`clamp_min=-1, clamp_max=1` 一致。輸出時 `adv_X = (adv_X / 2 + 0.5).clamp(0, 1)`。

遮罩為 $[0,1]$，閾值 0.5 二值化（與 PromptFlare 的「僅等於 1」不同）。
`masked_image = image * (mask < 0.5)`：mask=1 之處歸零，即 mask=1 為待重繪區。

## 5. img2img 是否有對應的 complex attack

**沒有。明確結論：repo 中 complex attack（diffusion attack）只有 inpainting 版本。**

證據三項：

1. `git/trees/main?recursive=1` 完整檔案清單中，`notebooks/` 只有四個
   `.ipynb`，其中唯一含 complex 的是 `demo_complex_attack_inpainting.ipynb`。
2. README 的章節結構：
   - 「Simple photo-guarding (Encoder Attack)」下有兩個子節：
     「Photo-guarding against Image-to-Image pipelines」→ `demo_simple_attack_img2img.ipynb`；
     「Photo-guarding against Inpainting pipelines」→ `demo_simple_attack_inpainting.ipynb`。
   - 「Complex photo-guarding (Diffusion attack)」下只列
     `demo_complex_attack_inpainting.ipynb`，並寫
     「For more effective photo-guarding **especially against image inpainting**」。
3. `demo_simple_attack_img2img.ipynb` cell 0 匯入的是
   `StableDiffusionImg2ImgPipeline`，全 notebook 只有 §2.3 的 encoder PGD，
   沒有 `attack_forward` / `compute_grad` / `super_l2` / `super_linf` 任一函式。

**對本專案的直接影響：** 本專案威脅模型為 img2img / SDEdit。若要以
PhotoGuard-c 作為 baseline，`attack_forward` 必須**自行改寫**為 img2img 前向
（加噪到 `strength` 對應的 $t$、無 mask 與 masked_image 的 9 通道拼接、
latent 由 `vae.encode(x_adv)` 而非隨機噪聲初始化）。此改寫在 repo 中不存在，
任何「照 PhotoGuard 官方實作跑 img2img complex attack」的說法都無原始碼依據，
必須在論文中明確標註為我方移植版本並記錄改動。

另註：`demo_simple_attack_img2img.ipynb` cell 12 呼叫
`pipe_img2img(prompt=..., init_image=..., ...)`，使用的是舊 API 參數名
`init_image`（diffusers 0.10.2，見 `requirements.txt`）。新版 diffusers 已改為
`image`，直接跑會報錯。

### 5.1 編輯（評估）階段設定

| 參數 | img2img（simple notebook cell 12） | inpainting（complex notebook cell 14） | `demo/app.py` |
|---|---|---|---|
| prompt | `"dog under heavy rain and muddy ground real"` | `"two men in the plane hugging"` | 使用者輸入 |
| SEED | 9222 | 9209 | 1234 |
| strength | 0.5 | 0.7 | 未傳（用預設） |
| guidance_scale | 7.5 | 7.5 | 7.5（`app.py:24`） |
| num_inference_steps | 50 | 100 | 100（`app.py:25`） |
| eta | 未傳 | 1 | 1 |

論文附錄 Table 8 給的是 height 512 / width 512 / guidance_scale 7.5 /
num_inference_steps 100 / eta 1。img2img notebook 用 50 步，與 Table 8 不符。

## PhotoGuard 未找到的項目

1. **img2img 版的 complex / diffusion attack**：不存在（見 §5）。
2. **論文 Table 9 所述 L∞ / 16/255 / 2/255 / 200 steps 這組參數的可執行程式碼**：
   `demo/app.py` 的 encoder attack 用 `eps=0.12`（$[-1,1]$ 值域 ≈ 15.3/255 像素域）
   最接近，但 `iters=200` 而非 notebook 的 1000；complex attack 實際跑的是 L2。
   **未找到任何一處同時滿足 Table 9 全部四欄的程式碼。**
3. **`grad_reps` 在論文中的對應敘述**：ar5iv 全文中搜尋 `grad_reps` 無結果，
   Algorithm 2 亦無梯度平均步驟。此為只在程式碼中存在的實作細節。
4. **evaluation / metrics 腳本**：repo 中不存在。
5. **批次處理多張影像的腳本**：僅有單張 demo notebook 與 gradio app。
6. **`utils.py` 中的 `resize_and_crop`**：`demo/app.py:10` 匯入，
   `demo/utils.py` 中存在（本次未逐行引用）；`notebooks/utils.py` 中不存在。
7. **PhotoGuard 論文的 arXiv HTML（ar5iv）核對結果**（供對照，非程式碼）：
   - Table 8：height 512、width 512、guidance_scale 7.5、num_inference_steps 100、eta 1。
   - Table 9：「Norm $\ell_\infty$ / $\epsilon$ 16/255 / step size 2/255 /
     number of steps 200」，並註明「For both of the attacks, we use the same set
     of hyperparameters shown in Table 9」。
   - Eq. 4：$\delta_{encoder} = \arg\min_{\lVert\delta\rVert_\infty \le \epsilon}
     \lVert \mathcal{E}(x+\delta) - z_{targ} \rVert_2^2$。
   - Eq. 5：$\delta_{diffusion} = \arg\min_{\lVert\delta\rVert_\infty \le \epsilon}
     \lVert f(x+\delta) - x_{targ} \rVert_2^2$。
   - 正文：「we backpropagate through only a few steps of the diffusion process,
     instead of the full process」、「We used an A100 with 40 GB memory」。

---

# 附錄：程式碼與論文正文的差異彙整

| 項目 | 論文 | 程式碼 | 影響 |
|---|---|---|---|
| PromptFlare eps | 「PGD epsilon budget of 12/255」 | `12/255` 施加於 $[-1,1]$ → 像素域 6/255；sample 實測最大差 7/255 | 失真預算實為一半 |
| PromptFlare step | 「a step size of 2/255」 | 同上 → 像素域 1/255 | |
| PromptFlare 迭代 | 「one-step gradient averaging repeated over 400 iterations」 | `--epochs` 預設 400、`grad_reps = 1`、`k = 1` | 一致 |
| PromptFlare 層聚合 | Eq. 12 $\mathbb{E}_l[\cdot]$ | `text_losses += text_loss`（加總） | sign-PGD 下不影響方向 |
| PromptFlare CA 定義 | Eq. 10 $\mathcal{A}V$ | 記錄 $\mathcal{A}V W_{out}$（`to_out` 之後） | 影響數值，不影響概念 |
| PromptFlare 其他 | CFG 7.5 / 50 步 / strength 1.0 | `inpaint.py` 預設一致 | 一致 |
| PhotoGuard norm | Table 9 $\ell_\infty$ | complex attack 實跑 **L2**（`super_l2`, `maxnorm=16`） | 不同的失真幾何 |
| PhotoGuard eps | Table 9 16/255 | encoder: 0.06 或 0.12 於 $[-1,1]$；complex: L2 半徑 16 | 無法直接對應 |
| PhotoGuard 損失 | Eq. 4/5 用 $\lVert\cdot\rVert_2^2$ | 全部用 `.norm()`（未平方） | sign/normalize 更新下方向相同 |
| PhotoGuard target | 「gray image」 | complex 實跑為零張量（= $[-1,1]$ 的中性灰）；`app.py` 用灰階測試圖；simple img2img 無 target | 一致或等價 |
| PhotoGuard img2img-c | 論文未區分 | **不存在** | 需自行移植並記錄 |

---

# 附錄 B：`super_l2` 的 `maxnorm=16` 究竟約束什麼

## B.1 原始碼（`demo_complex_attack_inpainting.ipynb` cell 8）

```python
        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)
```

## B.2 張量形狀（實測）

`X` 即 `cur_masked_image`，來自 `notebooks/utils.py:28–42` 的
`prepare_mask_and_masked_image`，第 30 行 `image = image[None].transpose(0, 3, 1, 2)`
使其形狀為 **`[1, 3, 512, 512]`**（本機以相同程式碼實跑確認）。

## B.3 `torch.renorm(d_x, p=2, dim=0, maxnorm=16)` 的語意（本機實測）

`torch.renorm` 對 **`dim` 所指維度的每一個切片**分別歸一化。`dim=0` 且 batch = 1，
因此切片只有一個，即整張 `[3, 512, 512]` 張量。實測（torch 2.13.0）：

| 測試 | 結果 |
|---|---|
| `torch.renorm(randn(1,3,512,512)*10, p=2, dim=0, maxnorm=16).norm()` | **16.000** |
| 同上，逐通道 `out[0,c].norm()` | 9.237 / 9.252 / 9.224（各自 **不** 等於 16） |
| `torch.renorm(randn(2,3,8,8)*10, ...)` 逐切片 norm | 16.0 / 16.0（切片獨立） |

結論：**是整張圖（3×512×512 = 786,432 個元素）展平後的單一 L2 範數 ≤ 16**，
不是逐通道、不是逐像素。

## B.4 值域與像素域換算

值域為 **$[-1, 1]$**（`clamp_min=-1, clamp_max=1`；`prepare_mask_and_masked_image`
第 31 行 `/ 127.5 - 1.0`）。以下為算術換算，非原始碼直述：

| 量 | 值 |
|---|---|
| L2 半徑（$[-1,1]$ 值域） | 16 |
| L2 半徑（$[0,1]$ 值域） | 8 |
| 均攤到全部 786,432 元素的 RMS（$[-1,1]$） | $16/\sqrt{786432} = 0.01804$ |
| 同上換算為 0–255 灰階 | **2.30 levels** |

但 `compute_grad` 中 `grad = ... * (1 - cur_mask)` 使擾動只存在於非重繪區，
故實際 RMS 應以該區元素數計。以 notebook 使用的 `assets/trevor_5.tif`
實測（`ImageOps.invert` → resize 512 → 閾值 0.5）：

- 重繪區（`mask == 1`）佔 75.1%，可擾動區（`mask == 0`）佔 **24.9%**
- 可擾動元素數 = 195,792
- 該區 RMS = $16/\sqrt{195792} = 0.03616$（$[-1,1]$）→ **4.61 / 255 灰階**

## B.5 與 L∞ 預算的關係

L2 球與 L∞ 球無法互換：半徑 16 的 L2 球允許少數像素出現遠大於 4.61 灰階的
偏移（極端情形為單一元素偏移 16，即滿量程），也允許全域低幅擾動。因此
**「PhotoGuard-c 的 eps = 16」不能被讀成 16/255 的逐像素上限**，其失真幾何
與論文 Table 9 的 $\ell_\infty$ 16/255 是兩種不同的約束。要與本專案的
L∞ / LPIPS 綁定約束比較，必須明確聲明採用哪一種，並且不可直接引用論文
Table 9 的數字去描述 notebook 實際跑出來的結果。
