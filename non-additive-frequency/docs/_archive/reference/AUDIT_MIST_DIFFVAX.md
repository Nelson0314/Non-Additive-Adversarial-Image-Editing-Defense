# Mist (v1) 與 DiffVax 原始碼查證報告

查證日期：2026-08-05
查證方式：完整 clone 官方 repo 後逐行閱讀，並取論文全文核對。
所有數值均標明來源檔案與行號；查不到者一律列入「未找到的項目」，不作推斷。

| 對象 | Repo | commit | 取得時間 |
|---|---|---|---|
| Mist v1 | `https://github.com/mist-project/mist.git` | `99f5f3c455022bef77ef25ba440a4289ea472d25`（2025-05-30） | 2026-08-05 |
| DiffVax | `https://github.com/ozdentarikcan/DiffVax.git` | `77fe66a8e6d50223a929081ec0851d9a49153903` | 2026-08-05 |

---

## 對象一：Mist（arXiv:2305.12683）

### 0. 版本確認

`mist-project/mist` 即 arXiv:2305.12683 對應的 repo，證據有二：

1. 論文正文（PDF 第 1 頁，第 69–70 行）明寫：
   > "Mist is currently available on GitHub: `https://github.com/mist-project/mist`."
2. Repo `Readme.md:3`：
   > "Our paper is acceped by ICML 2023 as Oral presentation. The paper can be found at [arxiv](https://arxiv.org/abs/2302.04578) currently. Mist is based upon the paper with some extensions (See our [technical report](http://arxiv.org/abs/2305.12683) for more details)."

`psyker-team/mist-v2` 為後續版本，其 README 引用的是 arXiv:2310.04687（另一篇論文），與 2305.12683 不同方法；`mist-project/mist-v2` 位址回傳 HTTP 301，重導至 psyker-team 組織（`mist-project` 與 `psyker-team` 為同一團隊改名，`mist-project/mist` 的 HEAD commit 即為 merge from `psyker-team/caradryanl-patch-1`）。**本節所有內容均取自 v1 repo。**

主程式為 `mist_v3.py`（README 內文提到的 `mist_v2.py` / `mist_v2_vangogh.py` 在 repo 中不存在，README 已與程式碼脫節）。

### 1. Textural loss 與 Semantic loss 的完整計算式

兩者都在 `mist_v3.py` 的 `target_model` 類別中實作。

#### 1.1 共用的編碼函式

`mist_v3.py:108-120`：

```python
    def get_components(self, x, no_loss=False):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(device)
        c = self.model.get_learned_conditioning(self.condition)
        if no_loss:
            loss = 0
        else:
            loss = self.model(z, c)[0]
        return z, loss
```

其中 `get_first_stage_encoding` 定義於 `ldm/models/diffusion/ddpmAttack.py:541-548`：

```python
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
```

即 `z = 0.18215 * sample(q(z|x))`（`scale_factor: 0.18215` 見 `configs/stable-diffusion/v1-inference-attack.yaml`）。注意這裡用的是 **隨機 sample 而非 mode/mean**，因此 textural loss 每次前傳都帶有 VAE 後驗抽樣雜訊。

#### 1.2 Textural loss

距離函式定義於 `mist_v3.py:102`：

```python
        self.fn = nn.MSELoss(reduction="sum")
```

在 `forward`（`mist_v3.py:137-138, 147`）：

```python
        zx, loss_semantic = self.get_components(x, True)
        zy, _ = self.get_components(self.target_info, True)
        ...
            return self.fn(zx, zy)
```

即

```
L_textural = sum_over_all_elements( (0.18215*E_sample(x+δ) - 0.18215*E_sample(y))^2 )
```

`y` 為 target image（見第 4 節）。

**與論文的落差**：論文 Eq (3)（PDF 第 137 行）寫成 `‖E(y) − E(x+δ)‖₂`，是 L2 範數；原始碼用 `nn.MSELoss(reduction="sum")`，是 **平方 L2 且對所有元素求和**，即 `‖·‖₂²`。兩者在梯度方向上等價（單調變換），但在與 semantic loss 融合時尺度不同，會直接影響 `w` 的有效值。

#### 1.3 Semantic loss

`mist_v3.py:139-140`：

```python
        if self.mode != 1:
            _, loss_semantic = self.get_components(self.pre_process(x, self.target_size))
```

`pre_process` 為 `RandomCrop`（`mist_v3.py:122-127`）：

```python
    def pre_process(self, x, target_size):
        processed_x = torch.zeros([x.shape[0], x.shape[1], target_size, target_size]).to(device)
        trans = transforms.RandomCrop(target_size)
        for p in range(x.shape[0]):
            processed_x[p] = trans(x[p])
        return processed_x
```

（預設 `block_num=1`、`input_size=512`、輸入亦為 512×512 時，`RandomCrop(512)` 等同 identity。）

`loss = self.model(z, c)[0]` 呼叫 `LatentDiffusion.forward`，`ldm/models/diffusion/ddpmAttack.py:869-878`：

```python
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        ...
        return self.p_losses(x, c, t, *args, **kwargs)
```

`p_losses`（`ddpmAttack.py:1011-1044`）核心：

```python
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        ...
        elif self.parameterization == "eps":
            target = noise
        ...
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        ...
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        ...
        loss = self.l_simple_weight * loss.mean()
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss += (self.original_elbo_weight * loss_vlb)
```

預設 `logvar_init=0.`、`l_simple_weight=1.`、`original_elbo_weight=0.`（`ddpmAttack.py:64-72, 97-98, 111`），因此實際化簡為

```
L_semantic = E_{t~U(0,1000), ε~N(0,I)} [ mean_over_CHW( (ε − ε_θ(z_t, t, c))² ) ]
```

單次前傳只抽一個 `t` 與一個 `ε`（Monte Carlo，1 個樣本），與論文 Eq (2) 敘述一致。`loss_type` 預設為 `'l2'`（`ddpmAttack.py:108`、`get_loss` 於 `ddpmAttack.py:279-283`）。

條件 `c` 由 `get_learned_conditioning(self.condition)` 產生，`condition` 設定於 `mist_v3.py:177-193`：

```python
    imagenet_templates_small_style = ['a painting']
    imagenet_templates_small_object = ['a photo']
    ...
    if object:
        imagenet_templates_small = imagenet_templates_small_object
    else:
        imagenet_templates_small = imagenet_templates_small_style

    input_prompt = [imagenet_templates_small[0] for i in range(1)]
```

`__main__` 從未傳入 `object`，因此實際永遠是 **`'a painting'`**。文字編碼器為 `FrozenCLIPEmbedder`，`version="openai/clip-vit-large-patch14"`、`max_length=77`（`ldm/modules/encoders/modules.py:137-145`）。

### 2. 融合方式與權重

`mist_v3.py:141-149`：

```python
        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 0:
            return - loss_semantic
        elif self.mode == 1:

            return self.fn(zx, zy)
        else:
            return self.fn(zx, zy) - loss_semantic * self.rate
```

PGD 端以 `targeted=True` 呼叫（`mist_v3.py:255`）：

```python
    attack = LinfPGDAttack(net, fn, epsilon, steps, eps_iter=alpha, clip_min=-1.0, targeted=True)
```

而 `Masked_PGD.py:186` 把 `targeted` 直接當成 `minimize`：

```python
            loss_fn=self.loss_fn, minimize=self.targeted,
```

`Masked_PGD.py:68-71`：

```python
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss
```

`loss_fn` 是 `identity_loss`（`mist_v3.py:68-78`），`forward(x, y)` 直接回傳 `x`。因此整體目標是

```
δ* = argmin_δ [ L_textural(x+δ, y) − w · L_semantic(x+δ) ]
   = argmax_δ [ w · L_semantic(x+δ) − L_textural(x+δ, y) ]
```

與論文 Eq (4)（PDF 第 150–156 行）完全同號：
> `δ := arg max_δ (w E_{x'_{1:T}} L_DM(x', θ) − L_E(x, δ, y))`

**權重 w 的實際值**（`mist_v3.py:280`）：

```python
    rate = 10 ** (args.rate + 3)
```

`args.rate` 預設 1（`mist_utils.py:79-86`），故 **w = 10⁴ = 1e4**，與論文正文「default fused weight of 1e4」（PDF 第 201 行）一致。WebUI 同樣：`mist-webui.py:40` `config['parameters']["rate"] = 10 ** (rate + 3)`，slider 預設值 1（`mist-webui.py:93`）。

### 3. eps / step_size / steps

CLI 預設（`mist_utils.py`）：

| 參數 | 行號 | 預設值 |
|---|---|---|
| `--epsilon` / `-e` | 35-43 | `16` |
| `--steps` / `-s` | 44-52 | `100` |
| `--input_size` | 53-61 | `512` |
| `--block_num` / `-b` | 62-70 | `1` |
| `--mode` | 71-78 | `2` |
| `--rate` | 79-86 | `1` |

step size（`alpha`）**沒有 CLI 參數**，只在 `init()` 簽名有預設值（`mist_v3.py:152`）：

```python
def init(epsilon: int = 16, steps: int = 100, alpha: int = 1,
```

`__main__` 呼叫 `init(epsilon=epsilon, steps=steps, mode=mode, rate=rate)`（`mist_v3.py:288, 342`），未傳 `alpha`，故實際恆為 `alpha=1`。

數值換算（`mist_v3.py:197-205`）：

```python
    parameters = {
        'epsilon': epsilon/255.0 * (1-(-1)),
        'alpha': alpha/255.0 * (1-(-1)),
        'steps': steps,
        ...
    }
```

因為影像被映射到 `[-1, 1]`，乘上 `(1-(-1)) = 2` 之後，等價於 `[0,1]` 值域下的 `16/255` 與 `1/255`。

關於 17/255，`mist_v3.py:156-159` 的 docstring 明寫：

```python
    :param epsilon: Strength of adversarial attack in l_{\infinity}.
                    After the round and the clip process during adversarial attack,
                    the final perturbation budget will be (epsilon+1)/255.
```

即程式碼名目上是 `16/255`，作者宣稱經過取整與裁切後實際上界為 `(16+1)/255 = 17/255`，對應論文正文（PDF 第 195–198 行）：
> "we set the sampling step as 100, the per-step perturbation budget as 1/255 and the total budget as 17/255."

**與論文的落差**：程式碼中沒有任何一行實作「+1/255」；那只是 docstring 的解釋。程式碼實際的投影上界為 `batch_clamp(eps, delta.data)`（`Masked_PGD.py:76`），eps = 32/255（[-1,1] 值域）= 16/255（[0,1] 值域）。要嚴格重現 17/255，必須自行決定是否把 `-e` 設為 17。

其他 PGD 細節（`Masked_PGD.py:141-192`）：`rand_init` 預設 `True`，且呼叫端未關閉，因此 δ 有隨機初始化（論文正文未提及）；`clip_min=-1.0`、`clip_max` 使用預設 `1.`；`ord=np.inf`；更新規則為 `delta += eps_iter * sign(grad)`（`Masked_PGD.py:74-78`）。

### 4. Target image 的取得方式

`mist_v3.py:284`：

```python
    target_image_path = 'MIST.png'
```

該檔在 repo 根目錄，尺寸 1440×1440、mode RGB（實測）。內容為純黑底 + 純白「MIST」字樣密集平鋪（3 欄 × 8 列），只有黑白兩色、邊緣為硬邊直角，即論文所稱的「high contrast ratio and sharp canny」。

載入方式（`mist_v3.py:298-302`）：

```python
                tar_img = load_image_from_path(target_image_path, target_size[0],
                                               target_size[1])
            else:
                img = load_image_from_path(image_path, input_size)
                tar_img = load_image_from_path(target_image_path, input_size)
```

`load_image_from_path`（`mist_utils.py:141-156`）以 `PIL.Image.BICUBIC` resize 到與被保護影像同尺寸，再按 block 切塊（`mist_v3.py:316`）與被保護影像對齊。

論文 PDF 第 182–184 行：
> "Empirically, it is better to select images with high contrast ratio and sharp canny as the targeted image y."

論文 3.3 節（PDF 第 364–368 行）另外比較了 Zero Target / Target1（Sistine Chapel 雕塑照）/ Target2（結構性建築照）/ Target Mist 四種 target，**repo 中只提供 Target Mist（`MIST.png`）**，其餘三張未附。

### 5. SD 模型 id 與值域

模型：`Readme.md:25-31`

```
Official Stable-diffusion-model v1.4 checkpoint is also required, available at huggingface.
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir -p  models/ldm/stable-diffusion-v1
mv sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
```

程式碼預設路徑（`mist_v3.py:170-174`）：

```python
    if ckpt is None:
        ckpt = 'models/ldm/stable-diffusion-v1/model.ckpt'

    if base is None:
        base = 'configs/stable-diffusion/v1-inference-attack.yaml'
```

即 **CompVis Stable Diffusion v1.4 原始 `.ckpt`**（非 diffusers repo id）。config 走 `ldm.models.diffusion.ddpmAttack.LatentDiffusion`（為了打開梯度流而改寫自 `ddpm.py`，見 `Readme.md:84-89`）。

值域：`[-1, 1]`。`mist_v3.py:230-233`：

```python
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img[:, :, :3]
    if tar_img is not None:
        tar_img = np.array(tar_img).astype(np.float32) / 127.5 - 1.0
```

輸出時轉回 `[0,255]`（`mist_v3.py:260-261`）：

```python
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
```

### 6. 三種 mode 的切換與預設

程式碼端的定義（`mist_v3.py:96, 143-149, 164`）：

```python
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
```

```python
        if self.mode == 0:
            return - loss_semantic
        elif self.mode == 1:

            return self.fn(zx, zy)
        else:
            return self.fn(zx, zy) - loss_semantic * self.rate
```

WebUI 的對應（`mist-webui.py:23-28`）：

```python
    if mode == 'Textural':
        mode_value = 1
    elif mode == 'Semantic':
        mode_value = 0
    elif mode == 'Fused':
        mode_value = 2
```

**預設值為 2（Fused）**：`mist_utils.py:71-78` `--mode` 預設 `2`；`mist-webui.py:89` `gr.Radio([...], value="Fused")`。與論文正文（PDF 第 200 行）「The default mode for Mist is the fused mode」一致。

**Repo 內部矛盾（必須注意）**：`Readme.md:45` 的表格寫

```
| Mode | {0 (textural), 1 (semantic), 2 (fused)} |
```

這與程式碼相反。以 `mist_v3.py` 與 `mist-webui.py` 為準：**0 = semantic、1 = textural、2 = fused**。

### 7. 其他已核對的實驗設定（論文 vs. repo）

| 項目 | 論文正文（PDF 行號） | Repo 對應 |
|---|---|---|
| 資料集 | Van Gogh paintings from WikiArt（198-199） | `test/vangogh/` 內 20 張 png |
| GPU | NVIDIA RTX 3090（202） | 未在程式碼中出現 |
| 輸入轉換 robustness | crop 64 px 各方向後 resize 回 512×512（238-240） | `utils/postprocess.py:6-22`，`crop_resize_from_path(input_path, 512, 384)`，`crop=(512-384)//2=64` |
| textual inversion 設定 | 8 vectors/token、6000 steps（213-214） | 未提供 |
| dreambooth 設定 | lr 2e-6、max 2000 steps、重訓 UNet + text encoder（215-217） | 未提供 |

### 8. Mist：未找到的項目（逐條）

1. **`mist_v2.py` / `mist_v2_vangogh.py` 不存在**。`Readme.md:37, 51, 56` 指示執行這兩個檔案，repo 內只有 `mist_v3.py`。
2. **17/255 沒有對應程式碼**。只有 `mist_v3.py:156-159` 的 docstring 宣稱「(epsilon+1)/255」，程式的投影上界實為 `16/255`。
3. **step size 沒有 CLI 開關**，只能改 `init()` 的預設值 `alpha=1`；`--alpha` 參數在 `mist_utils.parse_args` 中不存在。
4. **`object` 旗標無法從命令列設定**。`init(object=...)` 存在，但 `__main__` 未傳，prompt 恆為 `'a painting'`。論文正文完全未提及 prompt 的選擇。
5. **論文中的 FID / precision 評測程式未附**。repo 沒有任何計算 FID、precision 的程式碼。
6. **textual inversion、DreamBooth、scenario.gg、NovelAI 的評測 pipeline 未附**。
7. **Zero Target / Target1 / Target2 三張 target image 未附**，只有 `MIST.png`。
8. **論文未給定 target image 的解析度與對齊方式**；repo 的作法（BICUBIC resize 到與輸入同尺寸後逐 block 對齊）僅存在於程式碼。
9. **論文未說明 PGD 是否使用 random init**；程式碼使用（`rand_init=True` 預設未關閉）。
10. **論文未說明 textural loss 用的是 L2 還是平方 L2**（正文寫 `‖·‖₂`，程式碼是 `MSELoss(reduction="sum")`）。
11. **論文未說明 VAE 編碼是取樣還是取均值**；程式碼取樣（`encoder_posterior.sample()`）。
12. **`Readme.md` 的 mode 對照表與程式碼相反**（見第 6 節），無法判定何者為作者本意，只能以可執行的程式碼為準。
13. **未找到官方對「(epsilon+1)/255」推導的說明文件**（`mist-documentation.readthedocs.io` 未在本次查證中取用）。

---

## 對象二：DiffVax（ICLR 2026，arXiv:2411.17957）

### 1. 損失函式的完整定義與 α 的實際值

論文（arXiv HTML 版）給出：

```
L = α · L_noise + L_edit
L_noise = (1/sum(M)) · ‖(I_im − I) ⊙ M‖_p      （文中稱 p=1 為經驗最佳）
L_edit  = (1/sum(~M)) · ‖SD(I_im, ~M, P) ⊙ (~M)‖_1
```

原始碼實作於 `src/diffvax/immunization/diffvax_immunization.py:140-159`：

```python
                img_f = img_batch.float().cuda()
                unet_out = self.unetmodel.forward(img_f)

                unet_out = unet_out.half().cuda() * (1 - mask_batch)
                img_adv = torch.clamp(
                    img_batch + unet_out, self.clamp_min, self.clamp_max
                )
                img_out = self.attack_model.attack(
                    prompt=prompt_batch,
                    masked_image=img_adv,
                    mask=mask_batch,
                    num_inference_steps=4,
                    batch_size=batch_size,
                )

                target_image = torch.zeros_like(img_out).cuda()

                loss1 = (((img_out - target_image) * (mask_batch / 512)).norm(p=1) / (mask_batch / 512).sum())
                loss2 = (alpha * (img_adv - img_batch) * ((1 - mask_batch) / 512)).norm(p=1) / ((1 - mask_batch) / 512).sum()
                loss = loss1 + loss2
```

對應關係（**注意 mask 的命名與論文互補**）：

- 程式的 `mask_batch` = 論文的 `~M`（編輯區域）。`loss1` = `L_edit`。
- 程式的 `1 - mask_batch` = 論文的 `M`（免疫區域，加擾動處）。`loss2` = `α · L_noise`。
- `target_image = torch.zeros_like(img_out)` 使 `loss1` 化簡為 `‖SD(...) ⊙ (~M)‖₁ / sum(~M)`，與論文的 `L_edit` 一致（`vae.decode` 輸出值域為 `[-1,1]`，故 0 對應中灰）。
- 除以 512 在分子分母同時出現，數學上互相消去（L1 範數為一階齊次），只是 fp16 的數值縮放。
- `α` 寫在 L1 範數內部，因 `α>0` 與寫在外部等價。

**α 的實際值 = 4**，`configs/train.yml`：

```yaml
# Training
iter_num: 1000000
learning_rate: 0.00001
batch_size: 5
alpha: 4
```

`scripts/train.py:28, 91` 把它讀出並傳入 `train_immunization_all_images_batch(..., alpha=alpha, ...)`。函式簽名的預設值則是 `alpha=1`（`diffvax_immunization.py:91`），實際訓練走 config 的 4。

`loss_type` 參數（`diffvax_immunization.py:92` `loss_type="l2"`）**只用於檔名字串**（`diffvax_immunization.py:111, 116`），不影響任何計算；實際兩項損失都是 `p=1`。

### 2. Immunizer 網路架構、輸入與輸出

`src/diffvax/model.py:66-134` 的 `NestedUNet`（UNet++），檔頭註明改自 `https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py`。

- 建構：`NestedUNet(num_classes=3)`（`diffvax_immunization.py:54`），`input_channels` 預設 3、`deep_supervision=False`。
- 濾波器深度 `nb_filter = [32, 64, 128, 256, 512]`（`model.py:70`），下採樣 `MaxPool2d(2,2)`，上採樣 `Upsample(scale_factor=2, mode='bilinear', align_corners=True)`（`model.py:74-75`）。
- 基本單元 `VGGBlock` = Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU（`model.py:7-25`）。
- 輸出層 `self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)`（`model.py:103`），輸出 3 通道，**無任何 activation 或範圍限制**。
- 實測參數量：載入 `checkpoints/diffvax_trained.pth` 統計，共 212 個 tensor、**9,170,721 個參數**（fp32），與 README 宣稱的 ~9.2M 相符。

**輸入不是原圖，而是 masked image**——這是與論文敘述的落差。`scripts/train.py:72-76`：

```python
        mask_torch, image_torch, non_masked_image_torch = prepare_mask_and_masked_image(
            image, image_mask
        )
        image_torch = image_torch.half().cuda()
        non_masked_image_torch = non_masked_image_torch.half().cuda()
```

而 `src/diffvax/utils.py:30-45` 的回傳順序是 `(mask, masked_image, image)`：

```python
def prepare_mask_and_masked_image(image, mask):
    """Prepare image and mask tensors for inpainting."""
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    ...
    masked_image = image * (mask < 0.5)

    return mask, masked_image, image
```

因此 `image_torch`（進入 UNet++ 的張量）是**編輯區域已被歸零的影像**，`non_masked_image_torch`（完整原圖）在 `train.py` 中被載入後**從未使用**。`scripts/demo.py:288-294` 推論路徑同樣傳入 masked image。論文則寫「The immunizer model f(⋅;θ) takes an input image I」。這代表 **DiffVax 在免疫階段就必須事先知道編輯遮罩**。

輸出的施加方式（`diffvax_immunization.py:72-80`）：

```python
    def immunize_img(self, img, img_mask, epsilon=32):
        """Apply immunization perturbation to image."""
        img_f = img.float().cuda()
        unet_out = self.unetmodel.forward(img_f)
        unet_out = unet_out.half().cuda() * (1 - img_mask)

        img_adv = torch.clamp(img + unet_out, self.clamp_min, self.clamp_max)

        return img_adv, unet_out
```

即 **加性擾動**，只加在遮罩外，最後 clamp 到 `[-1, 1]`。`self.eps = 32 / 255` 與 `self.step_size = 1`（`diffvax_immunization.py:48-49`）以及 `immunize_img` 的 `epsilon=32` 參數**在整個 DiffVax 路徑中從未被使用**（僅 PhotoGuard / DiffusionGuard baseline 才有 PGD 預算），因此**沒有明確的 L∞ 上界約束**，只有 `α·L_noise` 這個軟性懲罰。

### 3. 訓練設定

| 項目 | 值 | 來源 |
|---|---|---|
| 資料集 | `ozdentarikcan/DiffVaxDataset`（HF），train 800 張 / validation 200 張，512×512 PNG，附 mask 與 `metadata.jsonl` 的 prompt | `configs/train.yml`、`scripts/train.py:130-135`、`src/diffvax/utils.py:95-137`、HF dataset card |
| 論文所述來源 | 「1000 images of individuals from the CCP dataset」，800 張訓練；mask 由 SAM 產生；prompt 由 ChatGPT 產生 | arXiv HTML |
| `iter_num` | `1000000` | `configs/train.yml` |
| batch size | `5` | `configs/train.yml` |
| optimizer | Adam | `diffvax_immunization.py:57` |
| learning rate | `0.00001`（1e-5） | `configs/train.yml`、`train.py:36` |
| 混合精度 | `torch.cuda.amp.GradScaler()`，模型/影像走 fp16 | `diffvax_immunization.py:15, 171-174`；`train.py:75-77` |
| 亂數種子 | `SEED=5` | `diffvax_immunization.py:89, 94` |
| 訓練時攻擊模型的採樣步數 | `num_inference_steps=4` | `diffvax_immunization.py:151` |
| 評測時的編輯步數 | `num_inf=30`，`guidance_scale=7.5`，`strength=1.0`，`eta=1` | `diffvax_immunization.py:204, 209-221`；`scripts/evaluate.py:24` |

`iter_num` 在程式碼中是 **epoch 數**（`diffvax_immunization.py:124` `for epoch_i in range(iter_num):`），README 也稱其為「Number of training epochs」。設為 `1000000` 表示實際上是跑到人為中止為止，**不是可重現的訓練終止條件**。

`train.py:79-83` 顯示每張影像的每個 prompt 都會展開成一筆訓練樣本，因此實際樣本數 = 800 × 每張的 prompt 數。

### 4. 預訓練權重與綁定的 SD 版本

- 權重：`checkpoints/diffvax_trained.pth`，36,782,602 bytes，212 個 tensor、9,170,721 參數、fp32，鍵名與 `NestedUNet` 完全對應（`conv0_0.conv1.weight` … `final.bias`）。**已隨 repo 提供，不需另外下載。**
- 綁定模型：`runwayml/stable-diffusion-inpainting`，出現在
  - `configs/train.yml`：`attack_model_link: "runwayml/stable-diffusion-inpainting"`
  - `scripts/demo.py:220`、`scripts/compare_baselines.py:180`：`default="runwayml/stable-diffusion-inpainting"`
  - `app.py:27`：`MODEL_ID = "runwayml/stable-diffusion-inpainting"`
- Pipeline：`StableDiffusionInpaintPipeline.from_pretrained(model_link, torch_dtype=torch.float16)`，scheduler 預設換成 `DDIMScheduler`（`src/diffvax/attack.py:15-23`）。
- 論文說主要針對 Stable Diffusion v1.5，並額外評測 SD v2 的跨模型遷移；**repo 只綁 SD 1.5 inpainting**。

### 5. 是否支援 img2img / SDEdit

**不支援。只支援 inpainting。** 證據：

- 全 repo（`*.py`、`*.yml`、`*.md`）對 `img2img`、`SDEdit`、`StableDiffusionImg2Img`、`InstructPix2Pix`、`ControlNet` 的搜尋結果為 **0 筆**（notebook 的 source cell 亦為 0 筆）。
- `src/diffvax/attack.py` 只 import 並實例化 `StableDiffusionInpaintPipeline`（`attack.py:3-19`）。
- `attack()` 的可微分前傳硬編碼了 inpainting 的 9 通道輸入（`attack.py:94-97`）：

```python
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat(
                [latent_model_input, mask, masked_image_latents], dim=1
            )
```

- 訓練與推論一律需要 `mask`（見第 2 節），沒有 mask 就無法產生擾動。

論文正文有提到用 InstructPix2Pix 與 MagicBrush 做 instruction-based editing 的評測，但**這些程式碼不在 repo 內**。

### 6. Counter-attack 評測（CNN 去噪、JPEG 0.75、IMPRESS）的實作位置

**未找到。repo 內完全沒有 counter-attack 的實作。**

搜尋 `jpeg`、`impress`、`purif`、`denois`、`counter`、`compress`（`*.py`、`*.yml`、`*.md`，並另外解析 notebook 的 source cell）的全部命中如下：

```
./README.md:26  （摘要文字中的 "robust against counter-attacks"）
./src/diffvax/immunization/diffusionguard_immunization.py:17,47,54,80,87,89  （scipy 的 gaussian_filter，用於 DiffusionGuard baseline 的輪廓遮罩擴增，與 counter-attack 無關）
```

`scripts/evaluate.py` 只計算 PSNR / SSIM（noise 端）與 PSNR / SSIM / FSIM / CLIP score（edit 端），沒有任何前處理攻擊（`evaluate.py:28-33`）：

```python
    noise_metrics["psnr"] = create_metric(MetricType.PSNR)
    noise_metrics["ssim"] = create_metric(MetricType.SSIM)
    edit_metrics["psnr"] = create_metric(MetricType.PSNR)
    edit_metrics["ssim"] = create_metric(MetricType.SSIM)
    edit_metrics["fsim"] = create_metric(MetricType.FSIM)
    edit_metrics["clip"] = create_metric(MetricType.CLIP, model='ViT-B-32', pretrained_on='laion2b_s34b_b79k')
```

`src/diffvax/metrics/` 下只有 `psnr.py`、`ssim.py`、`fsim.py`、`clip_score.py`、`base.py`、`factory.py`。

論文端可取得的敘述僅有：
> "DiffVax is robust to common counter-attacks, including CNN-based denoising, JPEG compression, and IMPRESS... DiffVax consistently outperforms PhotoGuard-D across all scenarios."

以及 Table 2 的欄位說明「denoiser (D.), JPEG (compression ratio of 0.75) counter-attacks」。**去噪器的具體型號、權重來源、JPEG 的實作函式庫、IMPRESS 的超參數，論文與 repo 都沒有給。**

### 7. DiffVax：未找到的項目（逐條）

1. **counter-attack 評測程式完全不存在**（CNN denoiser、JPEG 0.75、IMPRESS 都沒有）。
2. **CNN denoiser 的網路型號與權重來源未指明**（論文只寫 "CNN-based denoiser"）。
3. **IMPRESS 的呼叫方式與超參數未指明**。
4. **img2img / SDEdit / InstructPix2Pix / MagicBrush 的評測程式不存在**，儘管論文報告了 InstructPix2Pix 與 MagicBrush 的結果。
5. **SD v2 的跨模型評測程式不存在**（repo 只綁 `runwayml/stable-diffusion-inpainting`）。
6. **影片保護（論文摘要宣稱「for the first time, effectively protects video content」）的程式碼不存在**。
7. **論文未給出 α 的數值**；只有 `configs/train.yml` 的 `alpha: 4`，無法確認這是否即為論文表格所用的值。
8. **論文未給出 learning rate、batch size、epoch 數、optimizer 細節**（只在附錄引用了 Adam 的論文）。
9. **訓練終止條件未定義**：`iter_num: 1000000` 是 epoch 數，程式沒有 early stopping、沒有 validation-based 選點，`torch.save` 只在跑完全部 epoch 或 loss 變 NaN 時觸發（`diffvax_immunization.py:185-195`）。因此 `checkpoints/diffvax_trained.pth` 對應的實際訓練步數**無從得知**。
10. **論文說 immunizer 吃完整影像 I，程式碼吃的是 masked image**（第 2 節），論文未說明這點。
11. **沒有 L∞ 或任何硬性失真預算**；`self.eps = 32/255`、`self.step_size = 1`、`immunize_img(epsilon=32)` 都是死參數，從未被引用。論文亦未宣告硬性預算。
12. **`loss_type="l2"` 是誤導性命名**，實際兩項損失都是 `p=1`，該字串只進檔名。
13. **HF dataset card 未說明來源資料集**；論文說是 CCP dataset，但 dataset card 沒有標示，也沒有授權/來源說明。
14. **訓練用的 `num_inference_steps=4` 未在論文中出現**；論文沒有說明訓練時 SD 前傳只跑 4 步。
15. **未找到 `image_prompt_json: "image_prompt_pairs_with_validation.json"` 這個檔案的使用路徑**——`configs/train.yml` 有此欄位，但 `scripts/train.py` 與 `utils.get_train_val_image_prompt_list` 一律改讀 `train/metadata.jsonl` 與 `validation/metadata.jsonl`，該 config 欄位是死設定。
16. **論文的 Table 2 數值無法從 repo 重現**（缺 counter-attack 與跨模型程式）。

---

## 附：兩者的重現風險摘要（供實驗設計參考）

- Mist 的 `L_textural` 是 **sum-reduction 的平方 L2**，與 `L_semantic` 的 **mean-reduction MSE** 尺度差距極大（512×512 影像下潛在空間 4×64×64 = 16384 個元素）。`w = 1e4` 這個數值只有在這組特定 reduction 下才成立，改動任一邊的 reduction 就必須重新調 `w`。
- Mist 的兩項損失每次前傳都含兩重隨機性（VAE 後驗抽樣、擴散時間步 `t` 與雜訊 `ε`），單步梯度雜訊大；100 步 PGD 的結果對 seed 敏感（`seed_everything(23)`，`mist_v3.py:154, 176`）。
- DiffVax 沒有硬性失真預算，「匹配人眼可辨失真」的比較必須自行加約束或掃 `α`，不能直接用預設 `α=4` 的權重。
- DiffVax 的 immunizer 需要編輯遮罩作為輸入的一部分（吃 masked image），且輸出擾動只加在遮罩外，這使它**無法直接作為 img2img/全域文字引導編輯的防禦基準**；要當 baseline 必須先決定如何定義遮罩（例如全 0 遮罩），而該設定不在原作者的訓練分布內。
