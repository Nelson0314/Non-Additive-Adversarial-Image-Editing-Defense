# `src/purify/` 與 `src/models/` 的缺陷審查

> 2026-08-05。審查者立場：找出會使實驗結果錯誤的問題，不評風格。
> 本次審查未修改任何檔案。
>
> 核對依據：`docs/_audit_purify.md`、`docs/SOURCE_AUDIT_2026-08-05.md` §8–§9、
> `docs/ARCH_2026-08-05.md` §7.1／§8，以及本機環境的實際執行
> （`C:/Users/nelso/miniconda3/envs/wacv/python.exe`；diffusers 0.39.0、
> transformers 5.14.1、torch 2.13.0+cu126、opencv 5.0.0 含 `ximgproc`、
> `piq` 已安裝、**`lpips` 未安裝**）。
> `tests/test_purify_new_ops.py` 與 `tests/test_sdxl.py` 實跑：**61 passed**。

**結論先講：沒有找到「致命」等級的缺陷**（結果錯且看不出來）。
找到 3 項「重要」與 6 項「次要」。其中最需要處置的是 §2.1
（SDXL 的無條件嵌入取錯，無症狀）與 §1.1（淨化評測會整段中止）。

審查重點清單中，以下六項**逐項核對後未發現缺陷**，理由列於 §3：
直通估計方向、Adverse Cleaner 的七個參數與 BGR／uint8 位置、
`resize_only` 與 DiffPure 的降升取樣一致性、CNN 去噪的命名與免責文字、
淨化輸出值域、以及「不換 attention processor／不載入替代 VAE」兩條禁令。

---

## 1. 淨化側

### [重要] 1.1 主驅動未依 `available` 篩選，E3 淨化評測會在第 7 個算子當場中止並丟失已收集的資料

**位置**：`scripts/run_defense.py:259-269`；`src/purify/ops.py:372-394`、`274-284`

**現況**：

```python
# scripts/run_defense.py:259
for kind, plist in eval_sweep().items():
    for pur in plist:
        xp = pur.evaluate(x_def)
        ...
        rows.append(row)
...
return rows            # 第 275 行，迴圈全部跑完才回傳
```

`eval_sweep()` 在此**不帶 `sd`**，且全庫沒有任何一處讀取 `Purifier.available`
（`grep -rn "\.available" src/ scripts/` 只命中 docstring）。
`eval_sweep()` 回傳的字典依插入順序迭代：

`blur → jpeg → noise → quantize → crop_resize → adverse_cleaner →`
**`cnn_denoise_substitute`（拋 NotImplementedError）** `→ impress（拋 ValueError）`
`→ diffpure（拋 NotImplementedError）→ resize_only`

**應為**：`src/purify/ops.py:380-381` 自己寫的契約——

```
呼叫端可先以 `Purifier.available` 篩選，未篩選時會在該算子上明確拋出而不是靜默略過。
```

唯一的呼叫端沒有做這件事。

**後果**：`evaluate()` 在 `cnn_denoise_substitute` 上拋出，而 `rows` 是函式區域
變數、只在第 275 行回傳，例外一起穿透到 `scripts/run_defense.py:704` 的呼叫點
（該處無 try/except），整個 run 中止。已跑完的 blur／jpeg／noise／quantize／
crop_resize／adverse_cleaner 六組結果**全部丟失**，且 `resize_only`——DiffPure
解析度歸因的唯一對照（`SOURCE_AUDIT` §9 第 4 項）——排在拋出點之後，永遠不會被量到。
淨化軸是本論文主張的評測軸，這使該軸目前無法產出任何資料。

**確信度**：確定（程式路徑可靜態判定；`available` 無任何呼叫端已以 grep 確認）。

---

### [重要] 1.2 `Purifier.available` 對 IMPRESS 只檢查 `sd`，不檢查 LPIPS 後端；本機環境下 `available` 為真而 `evaluate` 會拋出

**位置**：`src/purify/ops.py:274-284`；`src/purify/impress.py:54-69`

**現況**：

```python
@property
def available(self) -> bool:
    """相依是否齊備。False 表示呼叫 `forward`／`evaluate` 會拋出。"""
    ...
    if self.kind == "impress":
        return self.options.get("sd") is not None
```

`impress_real` 的預設 `backend="lpips"`，而 `_make_lpips` 在 `import lpips` 失敗時
拋 `NotImplementedError`。本機環境**沒有安裝 `lpips`**——證據是
`tests/test_purify_new_ops.py:239` 的
`test_impress_未安裝_lpips_套件時不得靜默改用他者` 在本次執行中**沒有 skip 而是通過**
（該測試在 `lpips` 存在時會 `pytest.skip`）。

**應為**：`available` 應與 `has_guided_filter()`／`has_diffpure_weights()` 同級，
一併檢查 LPIPS 後端是否可載入。

**後果**：即使 §1.1 被修好（呼叫端加上 `available` 篩選），IMPRESS 仍會通過篩選
然後在執行時拋出，篩選形同虛設。另有一層：`main_set(sd=...)` 與 `eval_sweep(sd=...)`
都未傳 `backend`，故永遠走 `lpips` 後端；若有人為了讓它跑起來改成 `backend="piq"`，
依 `impress.py:22-26` 的說明那是**我方替代的 LPIPS 實作**，必須在報表標註，
而目前沒有任何欄位承載這個標註。

**確信度**：確定（環境與測試行為皆已實測）。

---

### [重要] 1.3 `eval_sweep()` 在主驅動中不帶 `sd`，IMPRESS 永遠拿不到 VAE

**位置**：`scripts/run_defense.py:259`

**現況**：`for kind, plist in eval_sweep().items():`——`eval_sweep(sd=None)`。

**應為**：`eval_sweep(sd=sd)`。`evaluate()` 的簽名第一個參數就是 `sd`，取得不成問題。
`tests/test_purify_new_ops.py:346-348` 已釘住「傳了 `sd` 才會進 options」，
但沒有任何測試釘住呼叫端有傳。

**後果**：IMPRESS 這一格永遠是
`ValueError: IMPRESS 需要 SD 的 VAE`。IMPRESS 是 `SOURCE_AUDIT` §8 判定「可忠實實作」
的兩個算子之一，缺它會讓淨化組只剩演算法類（JPEG／crop／bilateral），
「非加性比加性更耐淨化」的主張少掉最強的一個對手。

**確信度**：確定。

---

### [次要] 1.4 `proxy_gap` 在現行設計下對所有確定性算子恆等於 0，無法作為「代理與真實一致」的證據

**位置**：`src/purify/ops.py:315-327`；`scripts/run_defense.py:240`、`267-268`

**現況**：

```python
def forward(self, x):
    if self.differentiable:
        return self._real(x)
    return straight_through(x, self._real(x))

def proxy_gap(self, x) -> float:
    with torch.no_grad():
        return float((self.forward(x) - self.evaluate(x)).abs().max())
```

`straight_through(x, hard) = hard.detach() + (x - x.detach())`，在 `no_grad` 下
`x - x.detach()` 逐元素精確為 0，故 `forward(x)` 與 `evaluate(x)` **位元等同**。
`forward` 走的就是 `_real`，兩者不可能不同。因此對每一個確定性算子，
`proxy_gap` 恆為 `0.0`——這不是量測結果，是恆等式。

CSV 側同樣：`run_defense.py:240` 先寫死 `"proxy_gap": 0.0`，
第 267-268 行只對不可微者以 `pur.proxy_gap(x_def)` 覆寫，而該值仍恆為 0。

**應為**：`src/purify/ops.py` 模組 docstring 第 9-11 行宣稱

```
代理與真實實作的差距必須在報告中明列，不得省略（spec §5.1 末段）。
`Purifier.proxy_gap` 提供直接量測此差距的方法，使該聲明有數字支撐而非只是免責聲明。
```

真正的差距在**梯度**（`jpeg_proxy` 的 docstring 自己講得很清楚：
「此代理的梯度是錯的，而非近似的」），而 `proxy_gap` 量的是前向。

**後果**：報表若出現一整欄 `proxy_gap = 0` 並被解讀為「訓練看到的與評測看到的
是同一件事」，那是錯的結論——兩者的前向本來就一樣，差別全在梯度，
而該差別完全沒有被量。這不會讓數字錯，但會讓一個免責聲明被誤當成已驗證。
建議改量可比較的東西（例如以有限差分近似的真實 Jacobian 對恆等映射的偏離），
或直接把該欄改名為「前向一致性檢查」並在報告中說明其恆為 0 的原因。

另一個副作用：`proxy_gap` 會把 `_real` 再跑兩次。Adverse Cleaner 實測
256² 需 0.15 s（本機 CPU，64 次 bilateral + 4 次 guided），512² 約 0.6 s、
1024² 約 2.4 s——每張圖每個算子多付兩倍成本，只為得到一個恆等於 0 的數。

**確信度**：確定（數學上可證，且已由 `tests/test_purify_new_ops.py:310` 反向印證）。

---

### [次要] 1.5 `resize_only` 與 DiffPure 的目標解析度是絕對值 256，同名欄位在 512² 與 1024² 下不是同一個算子

**位置**：`src/purify/diffpure.py:41`、`46-64`；`scripts/run_defense.py:334`（`--size` 預設 512）

**現況**：`DIFFPURE_RESOLUTION = 256`，`resize_roundtrip` 一律降到 256²。

**應為**：這是我方指定的處置（`SOURCE_AUDIT` §9 第 4 項），本身沒錯。
問題是沒有記錄它的解析度相依性：`--size 512` 時是 2× 降取樣，
SDXL 的 1024² 是 **4×** 降取樣，破壞力差很多。

**後果**：`purify=resize_only` 與 `purify=diffpure` 這兩列在 SD v1.x（512²）與
SDXL（1024²）之間**不可比**。這與 `attention.py:398-404` 已經記錄的
「`reduce="sum"` 跨模型不可比」是同一類問題，但淨化這邊沒有對應的註記。
若報表把兩批數字並列，resize 對照的結論會被解析度差異污染。

**確信度**：確定（程式）。

---

### [次要] 1.6 Adverse Cleaner 的輸入被量化到 uint8 網格，其他算子沒有；「Adverse Cleaner vs identity」的差值混入了 8-bit 量化

**位置**：`src/purify/adverse_cleaner.py:92`

**現況**：

```python
# 對應 cv2.imread：uint8 網格上的 [0,255] float32，通道為 BGR
img = np.round(rgb * 255.0).astype(np.float32)[:, :, ::-1].copy()
```

**應為**：這一步**忠於上游**（`clean.py` 的輸入來自 `cv2.imread`，本來就是 uint8），
轉寫沒有錯誤。但 `crop_resize`、`resize_only`、`impress`、`identity` 都在
float 上運作、不做這個量化，只有 `jpeg` 與 `quantize` 有。

**後果**：`identity → adverse_cleaner` 的差值同時包含「8-bit 量化」與
「64 次 bilateral + 4 次 guided」兩件事。若某個防禦的擾動有低於 1/255 的成分，
量化本身就會削掉它，而該效應會被記在 Adverse Cleaner 頭上。
`eval_sweep` 的 `quantize` 鍵已有 256 這一點，可直接作為分離用的對照，
但目前沒有任何地方指出這兩列要一起讀。

**確信度**：高（機制確定；實際幅度取決於各方法的擾動振幅，需實測）。

---

### [次要] 1.7 測試中的一條恆真斷言與一條同義反覆

**位置**：`tests/test_purify_new_ops.py:283`、`302-307`

**現況**：

```python
# 283
assert "非 NTIRE 2023 冠軍" in cnn_denoise_substitute_real.__doc__ or True
```

`X or True` 恆為真，這一行不驗證任何東西。docstring 中「非 NTIRE 2023 冠軍」
這句話因此**沒有被任何測試釘住**（第 281 行釘的是例外訊息，第 284 行釘的是
docstring 中的「我方替代」四字，不是這一句）。

```python
# 302-307
def test_前向差距等於回報的_proxy_gap():
    for p in _deterministic_ops():
        measured = float((p.forward(x) - p.evaluate(x)).abs().max())
        assert p.proxy_gap(x) == measured, p.kind
```

`measured` 的算式與 `Purifier.proxy_gap` 的函式本體逐字相同，此測試只驗證
「同一段程式跑兩次結果一樣」。

**後果**：兩者都不會讓實驗結果錯，但會讓「測試通過數 61」高估實際覆蓋。
第 283 行是本次唯一找到的恆真斷言。

**確信度**：確定。

---

## 2. SDXL 側

### [重要] 2.1 `encode_text("")` 不是 SDXL 的無條件嵌入；stock SDXL base 用的是零張量

**位置**：`src/models/sd.py:604-638`（`SDXLWrapper.encode_text`）；
使用點 `src/defense/optimize.py:743`、`scripts/run_defense.py:211`、`294`、
`src/defense/linf_attack.py:369`

**現況**：全庫以 `sd.encode_text("")` 作為 CFG 的無條件嵌入與 prompt-free 的條件嵌入。
`optimize.py:737-740` 把這個作法寫成了理由：

```
**訓練期一律餵空 prompt**（`DESIGN` §2.1 的 prompt-free 設計）。... 空
prompt 的 CLIP 編碼是 CFG 的無條件嵌入，在攻擊方的任何 prompt 下都存在。
```

**應為**：SDXL base 的 pipeline config 帶 `force_zeros_for_empty_prompt = True`。
本機 diffusers 0.39.0 的
`pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py::encode_prompt`
原文（已直接讀取原始碼確認）：

```python
zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
    negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
```

即：使用者不給 negative prompt 時，stock SDXL 的無條件分支用的是
**`torch.zeros_like`**，`embeds`（2048 維序列）與 `pooled`（1280 維，
會進 `added_cond_kwargs["text_embeds"]`）**兩者都是零**，
不是空字串的 CLIP 編碼。SD v1.x 沒有這個旗標，故 `encode_text("")` 在 SD v1.x 上是對的
——這是移植時新產生的落差，正對應 `ARCH_2026-08-05.md` §8 第 2 項標為「高風險」的那一格。

反證這件事已被知悉：`tests/test_sdxl.py:296` 自己在建 tiny pipeline 時傳了
`force_zeros_for_empty_prompt=True`，但 `SDXLWrapper` 從頭到尾沒有讀過這個 config 欄位。

**後果**，兩層且都無症狀：

1. **評測端**：`run_defense.py:211-213` 用 `emb_u = sd.encode_text("")` 餵給
   `_eps_cfg`。以 `guidance_scale=7.5` 計算的 `eps_u + w·(eps_c − eps_u)`，
   其 `eps_u` 不是 stock SDXL 會產生的那一個，故模擬的攻擊方**不是 stock SDXL**，
   而威脅模型（`CLAUDE.md`、`ARCH` §7.1）要求它必須是。編輯結果會偏，
   而所有「抵抗文字引導編輯」的數字都是在這條鏈上量的。
2. **訓練端**：`optimize.py:743` 的 prompt-free 條件嵌入 `emb_cond`，其存在理由
   （「空 prompt 的 CLIP 編碼是 CFG 的無條件嵌入」）在 SDXL 上為假。
   防禦因此是針對一個攻擊方從不使用的嵌入在訓練。

修法是一行：SDXL 的無條件條件應為
`SDXLPrompt(torch.zeros_like(embeds), torch.zeros_like(pooled))`
（或依 `pipe.config.force_zeros_for_empty_prompt` 分派），並記 before/after。

**確信度**：高。程式路徑與 diffusers 原始碼已直接核對；
`stabilityai/stable-diffusion-xl-base-1.0` 的 `model_index.json` 帶
`force_zeros_for_empty_prompt: true` 屬公開已知，但本機無該權重，
未在真實 config 上逐位元確認——列入 §4。

---

### [重要] 2.2 平台停止門檻的「須重新校準」清單指到了一個已經不生效的常數，真正生效的那個沒有被列

**位置**：`src/defense/optimize.py:191-215`、`219-235`；
`src/models/attention.py:462-466`；`tests/test_sdxl.py:595-596`

**現況**：

```python
# optimize.py:90
DEFENSE_MONITOR = {
    "targeted_output": "edit_shift",
    "targeted_attn": "shared_mass",
}
# optimize.py:212
MONITOR_TOL = {
    "edit_shift": 1e-4,     # E23 實測平均 5.4e-4
    "attn_div": 1e-5,       # 2026-08-04 實測平均 1.6e-4
}
```

`attn_div` **已不是任何 `defense_mode` 的監看量**（`attn_mode="divergence"` 已被移除，
見 `optimize.py:876-879`）。而

- `src/models/attention.py:463-466`：「`optimize.py` 中 `attn_div` 的平台停止門檻
  （1e-5，於 SD v1.4 上實測）都綁在舊層數上，必須在段 0 重新量測」
- `tests/test_sdxl.py:595-596` GPU 待驗清單第 5 項：「`optimize.py` 的 `attn_div`
  平台停止門檻 1e-5 ... 層數變 4.4 倍後必須重新校準」

兩處都只點名 `attn_div`。唯一實際會被 `resolve_stop_tol` 取用的
`edit_shift = 1e-4`（`E23 實測`，SD v1.4／512²／site PF）**沒有任何 SDXL 重校要求**，
且 `resolve_stop_tol(None, "edit_shift")` 會靜默回傳它。

順帶一提，`attention_divergence` 與 `shared_token_mass` 都是**逐層取平均**
（`attention.py:324`、`objective.py:315`），故那兩個量本身其實對層數不敏感；
「綁在舊層數上」這個理由對 `attn_div` 而言也不準確。真正會隨模型改變的是
`edit_shift`（1024² 的 LPIPS 動態範圍與 512² 不同）。

**應為**：待校準清單應把 `edit_shift` 的 1e-4 列進去；
或讓 `resolve_stop_tol` 在偵測到非校準時的模型／解析度時拋出，
一如它對 `shared_mass` 的處置（`optimize.py:206-211` 刻意留的拋出點）。

**後果**：SDXL 上以 `targeted_output` 訓練（本輪的主要模式）時，
停止門檻沿用 512²／SD v1.4 的值。若新尺度下每步改善量比 1e-4 小，
`plateau_stop` 會在第一個觀察窗就回報「已收斂」，訓練被砍短，
而 CSV 上顯示的是 `stop_reason = 約束已啟動且 edit_shift ... 低於 1.0e-04`
——正是他們自己在 `optimize.py:200-202` 記載的失效形態，且沒有任何症狀。
反之若門檻過鬆則每格都跑滿上限，落入 E21–E23 §5.4 的問題。

**確信度**：確定（程式路徑與清單內容）／推測（偏差幅度需在 GPU 上實測）。

---

### [次要] 2.3 `CrossAttentionRecorder` 只在建構時核對層數，`with` 區塊內若放了 CFG 就會記兩份且無檢查

**位置**：`src/models/attention.py:206-217`、`255-267`

**現況**：層數核對只發生在 `__init__`：

```python
if verify_count and cfg is not None:
    self.expected = cross_attention_layer_count(cfg)
    if self.expected["total"] != len(self._layers):
        raise RuntimeError(...)
```

`__enter__` 清空 `maps`、`__exit__` 只卸 hook，**沒有核對 `len(self.maps)`**。

目前兩個呼叫端都安全：`optimize.py:900-903` 與 `linf_attack.py:317-322` 用的都是
`sd._eps(...)`（單支前向），故每次 `with` 恰好記 `expected` 筆。

**應為**：`__exit__` 應核對 `len(self.maps) == self.expected["total"]`。
`__init__` 的 docstring 自己說明了理由：「漏掉的症狀只是『目標值偏小』，
沒有任何錯誤訊息」——同樣的道理適用於多記。

**後果**：只要有人把 `_eps_cfg`（CFG 兩支前向）放進 `with rec:`，
`maps` 會變成 140 筆，`aggregate_token_attention` 會把**條件與無條件的注意力圖
一起相加**，`shared_token_mass` 則會把無條件那一半平均進去。
兩者都是靜默的錯誤結果。考慮到 §2.1 的修法可能會動到 CFG 相關的呼叫端，
這個缺乏防護的點值得先補上。

**確信度**：確定（程式路徑）；目前尚未被觸發。

---

### [次要] 2.4 `test_GPU上待驗項目清單` 以 `assert True` 結尾

**位置**：`tests/test_sdxl.py:578-598`

**現況**：整個測試函式的本體是一段 docstring 加 `assert True`。

**應為**：這是**刻意的設計**（docstring 已說明「列成測試而非只寫在報告裡，
是為了讓它跟著程式走」），不是疏漏。但它會被計入 `61 passed`。

**後果**：無實驗影響。列出只為在盤點「假通過」時不遺漏——
它與 §1.7 的 `or True` 性質不同：前者是刻意的清單，後者是壞掉的斷言。

**確信度**：確定。

---

## 3. 逐項核對後**未發現缺陷**的審查重點

| 審查重點 | 核對結果 |
|---|---|
| **直通估計方向** | `ops.py:96-105` 為 `hard.detach() + (x - x.detach())`，方向正確。`tests/test_purify_new_ops.py:310-314` 的檢查非同義反覆——寫反成 `x + (hard - x).detach()` 會在 float32 上留下非零差，該測試會失敗 |
| **Adverse Cleaner 參數** | 七個常數（64／5／8／8／4／4／16）與 `_audit_purify.md` §1.3 逐項相符；`guided_filter(img, y, ...)` 的 guide 固定為原圖而非上一次結果，正確 |
| **Adverse Cleaner 值域與通道** | 先 `×255` 再濾波、`sigmaColor` 不換算，正確（若換到 [0,1] 而不改 sigma 會差 255 倍）。RGB→BGR（`[:, :, ::-1]`）與轉回皆正確。附註：bilateral 的顏色距離是逐通道絕對差之和、guided filter 的 3×3 共變異數在同時置換下等變，故 BGR 這一步在數值上其實是恆等；寫了沒有壞處 |
| **Adverse Cleaner 的 uint8 位置** | 輸入 `np.round`（對應 PNG 存檔）、輸出 `.clip(0,255).astype(np.uint8)`（**截斷**），與 `clean.py` 最後一行一致。位置正確 |
| **`resize_only` 與 DiffPure 的降升取樣** | 兩者共用 `resize_roundtrip`（`diffpure.py:46-69`），參數不可能分岔；`resize_params()` 與 `tests/test_purify_new_ops.py:128-142` 已釘住。對照有效（唯一的補充見 §1.5） |
| **CNN 去噪的命名與免責** | 函式名 `cnn_denoise_substitute_real`、kind `cnn_denoise_substitute`、docstring「**我方替代，非 NTIRE 2023 冠軍模型**」、例外訊息「我方替代，非 NTIRE 2023 冠軍」——三處齊備。`KINDS` 不含 `ntire2023`，有測試釘住 |
| **淨化輸出值域** | 全部算子的輸出皆在 [0,1]：`noise`／`crop_resize`／`resize_only`／`impress` 顯式 clamp，`quantize` 先 clamp 再 round，`jpeg`／`adverse_cleaner` 經 uint8，`blur` 為正規化核的凸組合。`identity` 直接轉交（輸入若越界則輸出越界，但呼叫端一律餵 [0,1]） |
| **不得載入替代 VAE 權重** | 全庫 grep `fp16-fix`／`fp16_fix`／`madebyollin`／`AutoencoderKL.from_pretrained` 只命中 docstring、文件與該項禁令的測試本身。`_apply_precision`（`sd.py:95-105`）只做 `.to(dtype)`，不存在載入路徑。`tests/test_sdxl.py:182-201` 以 AST 剝除 docstring 後掃描，並鎖死 `SDXLWrapper.__init__` 的參數集合。**設計正確** |
| **fp16 → VAE fp32 由程式強制** | `resolve_precision`（`device.py`）是純函式且為唯一來源；`SDWrapper.__init__:64` 一律經它取得，`_apply_precision` 實際施加到子模組。`tests/test_sdxl.py:547-567` 驗到「模型上的 dtype」而不只是規則，並驗了 fp16 latent 進 fp32 VAE 的解碼路徑。bf16 全程半精度亦有測試。**是規則不是註解** |
| **兩個 text encoder 與 `added_cond_kwargs`** | `hidden_states[-2]` 串接成 2048、pooled 只取自第二個 encoder 的 `out[0]`——與本機 diffusers 0.39.0 `encode_prompt` 原始碼逐句比對相符（該處亦為 `pooled_prompt_embeds = prompt_embeds[0]`）。`time_ids` 為 6 維 `(h, w, 0, 0, h, w)`、由 latent × `vae_scale_factor` 反推，與 base（非 refiner）一致；`_check_added_cond_dim` 提前核對 `256×6 + 1280 = 2816`。`SDXLPrompt` 把 pooled 攤平成 checkpoint 的獨立張量引數，避免藏在 dataclass 中而靜默斷梯度——這一點設計正確且有測試 |
| **SDXL 應得 70 個 `attn2`** | `cross_attention_layer_count` 的公式已逐項驗算：SDXL down `2×2 + 2×10 = 24`、mid `tl[-1] = 10`、up（`reversed` 後）`3×10 + 3×2 = 36`，合計 **70**；SD v1.5 得 **16**。與 diffusers `UNet2DConditionModel` 內部的 `reversed_layers_per_block + 1`／`reversed_transformer_layers_per_block`／mid 取 `[-1]` 三條規則一致。`CrossAttentionRecorder` 在建構時核對掃描值與推導值，不符即拋出 |
| **正規化常數已標為須校準** | `attention_region_mask` 的 τ 以**峰值正規化**，語意跨模型不變，且該性質有 `tests/test_lo_protocol.py` 釘住；`aggregate_token_attention` 的 `reduce="sum"` 明寫跨模型不可比、提供 `reduce="mean"`；`shared_token_mass` 與 `attention_divergence` 逐層取平均。唯一的缺口是停止門檻，已列為 §2.2 |
| **不可換 attention processor** | `attention.py` 用 `register_forward_pre_hook(..., with_kwargs=True)`，只讀該層輸入後以該層自己的 `to_q`／`to_k`／`get_attention_scores` 另算一次，UNet 前向不受影響。全庫僅 `src/baselines/promptflare.py` 呼叫 `set_processor`——那是 PromptFlare 官方實作本身的要求，且 `PromptFlareContext.close()` 會逐一還原、`prepare()` 對非 512² 輸入直接拋出。**未污染主路徑** |
| **hook 的 head 平均** | `head_to_batch_dim` 的輸出佈局為 `b·H + h`，故 `reshape(B, H, Q, T).mean(dim=1)` 正確；`group_norm`／`norm_cross` 兩者皆依 diffusers `AttnProcessor.__call__` 的順序處理 |
| **BDIA 與排程** | SDXL 預設 `EulerDiscreteScheduler` 在本機 diffusers 0.39.0 確實持有 `alphas_cumprod`（已實測，長度 1000），而本封裝只讀該陣列、不呼叫 `scheduler.step`，故排程器種類不影響結果。`bdia_denoise` 的下行遞迴與 `bdia_inversion` 互為精確反解，代數無誤 |

---

## 4. 我無法確認的項目（需 GPU 或權重）

1. **`stabilityai/stable-diffusion-xl-base-1.0` 的 `model_index.json` 是否確實帶
   `force_zeros_for_empty_prompt: true`**（§2.1 的前提）。本機無該權重。
   diffusers 的程式分支已直接核對，缺的只是該旗標在官方 repo 的實際值。
   **驗法**：`StableDiffusionXLImg2ImgPipeline.from_pretrained(...).config.force_zeros_for_empty_prompt`。
2. **SDXL 原生 VAE 在 fp16 下確實溢位成黑圖**。`tests/test_sdxl.py` 的極小 VAE 是
   隨機權重、通道數只有 8，重現不了真實激活量級（該檔第 591-593 行已自陳）。
   規則被釘住了，現象沒有。
3. **BDIA 在 SDXL 上的來回誤差是否仍比 DDIM 小 2 個數量級以上**，以及
   **fp16 與 fp32 的一致性**。`_eps` 把狀態張量保持在 fp32，但 ε 本身在 fp16 下
   只有 10 bit 尾數；必須在 1024²、真實權重下 fp16／fp32 各跑一次比較。
4. **70 層注意力擷取的記憶體**。粗估：64² 那 24 層每層在 head 平均**前**是
   `(B·H, 4096, 77)`，10 heads 時單層約 12.6 MB；`optimize_crossattn` 的
   `maps_list` 會保留整個計算圖（該處刻意不開 checkpoint，見其 docstring），
   且長度為 `len(eot_pairs)`。V100 32 GB 下的上限必須實測，
   `purify_mode="all"` × 多個淨化算子時尤其。
5. **`edit_shift` 的平台停止門檻在 SDXL／1024² 下的正確量級**（§2.2）。
   需先跑一批不開 `stop_on_plateau` 的 run，量出每步實際改善率再定門檻。
6. **Adverse Cleaner 的量化效應幅度**（§1.6）。需要各方法在同一 τ 下的實際
   擾動振幅分佈才能判斷 8-bit 量化削掉多少。
7. **IMPRESS 在真實 SD VAE 上的行為**。現有測試用的是 1×1 卷積的
   `_StubVAE`，它不代表真實 VAE 的重建誤差（該檔第 205-206 行已自陳），
   故 1000 步 Adam 的收斂行為、fp16 下 `ADAM_EPS = 1e-5` 是否足夠、
   以及 LPIPS hinge 是否會被觸發，都未驗。另需先安裝 `lpips` 套件（本機缺）。
8. **DiffPure 與 CNN 去噪的真實行為**。兩者皆為 stub，缺檢查點
   （`256x256_diffusion_uncond.pt`、Restormer σ=50 權重）與套件
   （`guided_diffusion`／`torchsde`／`basicsr`）。介面已定案但一行都沒跑過。
   `SOURCE_AUDIT` §9 第 3 項的「替代對象確定為 Restormer 或改 NAFNet／SCUNet」
   仍待裁決。
