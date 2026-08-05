# 審查報告：`src/defense/` 與 `src/residual/` 的邏輯檢查

> 2026-08-05。審查對象為條件 N1／N2／N3 自身的方法實作，判準是「量到的東西是不是
> 我們以為的東西」。範圍：`src/defense/objective.py`、`src/defense/optimize.py`、
> `src/residual/{base,composite,site_apa,site_warp,site_weight,site_latent,lowrank}.py`、
> `src/utils/{calibration,cellid,progress}.py`、`src/defense/generator.py`
> 與對應測試。基準測試狀態：`518 passed / 1 skipped / 1 xfailed`（實跑）。
>
> 本報告不修改任何檔案。

---

## 摘要

- **致命 1 件**：APA 移植（N3）的階段二參數梯度恆為零，最佳化靜默不更新。
- **重要 6 件**：預設保真係數使 L∞ 取代 LPIPS 成為綁定約束；停止門檻沿用舊模型的校準值
  且無 context 檢查；階段一的實作與 `DESIGN` §4 的宣稱不符；`config_hash` 的必填鍵未涵蓋
  參數化容量；射線縮放無最終容差檢查；N1 的注意力目標取自無條件分支。
- **次要 7 件**：見 §3。

---

## 1. 致命

### [致命] `build_apa` 的 `init_std=0.0` 使 APA 階段二的梯度恆為零

**位置**：`src/residual/site_apa.py:89-101`（`init_std=0.0`）、
`src/residual/lowrank.py:122-124`（`U = randn(...) * init_std`、`V = zeros`）、
`src/defense/optimize.py:540-545`（無法攔截的零梯度檢查）

**現況**

```python
# site_apa.py:89-101
    latent = LatentResidual(
        steps=steps, ...
        # 初值為零：φ=0 必須逐位元等同未注入。LatentResidual 的低秩外積
        # 參數化在兩個因子都為零時梯度也為零，故其內部採「一半高斯、一半零」
        # 的安排，此處給 0 指的是整體殘差為零。
        init_std=0.0,
        seed=seed,
    )
```

```python
# lowrank.py:122-124
u0 = torch.randn(steps, channels, max_rank, height, generator=gen) * init_std
self.U = nn.Parameter(u0)
self.V = nn.Parameter(torch.zeros(steps, channels, max_rank, width))
```

**應為**：`init_std` 控制的正是「一半高斯」的那一半（U），V 恆為零。`LowRankResidual`
的 docstring（`lowrank.py:84-86`）明寫「初始化採 U ~ N(0, init_std)、V = 0，故初始 Δ = 0
（即 x_def = x），同時 ∂L/∂V ≠ 0 使梯度可流動」。傳 `init_std=0.0` 使 U 也為零，
於是 `Δ = einsum(U, V)` 的兩個因子同時為零，`∂Δ/∂U ∝ V = 0`、`∂Δ/∂V ∝ U = 0`，
兩者的梯度皆精確為零。註解所要迴避的失效，正是此處這行程式造成的。應保留預設
`init_std=0.02`（或任何正值）——Δ 的初值本來就已經是零，不需要再把 U 歸零。

**實測驗證**（本機 `wacv` env，非推論）：

```
U norm 0.0  V norm 0.0
gradU  0.0
gradV  0.0
```

**後果**：`optimize()` 走 N3 階段二時，`torch.optim.Adam` 每步都對零梯度做更新，
參數永遠停在初值，latent 注入的殘差恆為零。`run_stages` 的防護
（`optimize.py:540-545`「參數在 backward 之後沒有任何梯度」）**不會觸發**——`p.grad`
是零張量而不是 `None`，該檢查只判 `is not None`。因此：

1. N3 退化成「只有階段一（LoRA 保真對齊）」的條件，DESIGN §4 所定義的攻擊有效性階段
   完全沒有執行。
2. 訓練跑得完、逐步日誌有數字、`x_def` 有輸出、`grad_norm` 欄位是 `0.000e+00` 但那一欄
   在多階段報表中不會被單獨檢視。這是 A1 的同一種失效型態（零梯度、無症狀），
   只是成因由損失形式換成參數初始化。
3. N3 相對 baseline 的任何比較結果都不成立：量到的是 LoRA 重建 + 射線縮放的效果。

**確信度**：確定（已實測梯度為零）。

**旁證**：`build_apa` 在整個 repo 中**沒有任何呼叫端、也沒有任何測試**
（`grep -rn build_apa` 只命中 `site_apa.py` 自身與 `docs/CODE_2026-08-05.md`）。
`tests/test_composite.py:52-53` 的 `_apa_like()` 以同一組參數建構
（`init_std=0.0`），但該檔只斷言「φ=0 時逐位元等同」——見 §4 的假通過分析。

---

## 2. 重要

### [重要] `beta_linf` 預設 100.0 使 L∞ 取代 LPIPS 成為位移場條件的綁定約束

**位置**：`src/defense/objective.py:163-164`（`beta_linf=100.0`、`tau_linf=0.06`）、
`src/defense/optimize.py:164`（`stop_require_feasible=True`）、
`src/defense/optimize.py:327-341`（`active_constraint_keys` 收錄 `fid_pen_linf`）

**現況**

```python
# objective.py:163-164
    beta_linf: float = 100.0
    tau_linf: float = 0.06     # ≈ 15/255，對抗擾動文獻的常見上限量級
```

```python
# optimize.py:164
    stop_require_feasible: bool = True
```

**應為**：`DESIGN` §3.2 明定「**共同貨幣取 LPIPS(x_def, x)，不取 L∞**。理由是實測 L∞
對非加性不具鑑別力：site S 在 LPIPS 0.1077 時 L∞ 已達 0.5439」。`site_warp.py:34`
自己也寫「既有的 L∞ hinge 對本位置可能過嚴，該係數須另行檢視」。全部既有 run 的
`env.json` 記錄的是 `beta_linf: 0.0`（`runs/e15_*`、`runs/e21_*`、`runs/e23_*`）。
本輪的預設值卻是 100.0。

**後果**：以 DESIGN 自己記錄的實測值代入，位移場在主表的 τ_LPIPS = 0.20 上，
L∞ 已在 0.5 量級，`pen_linf = 0.5 − 0.06 = 0.44`，乘上 100 即 44，而
`pen_lpips` 在 τ 之內為 0。於是

1. `L_fid` 幾乎全部由 L∞ hinge 構成，最佳化被一道「非本輪共同貨幣」的約束節流，
   LPIPS 到不了 τ，**τ 不是綁定的約束**——這正是 `objective.py:96-100` 刪除
   `alpha_lpips` 所要修掉的那一件事，只是換成 L∞ 這道 hinge 來施力。
2. `active_constraint_keys(LossConfig())` 包含 `fid_pen_linf`
   （`tests/test_plateau_stop.py:124-126` 斷言如此），而 `stop_require_feasible=True`
   要求**全部**啟動中的約束在停止那一刻都 ≤ 1e-6。位移場的 `pen_linf` 結構上不可能歸零，
   故 `plateau_stop` **永遠不會回傳 True**，每一格都跑滿 `max_steps` 並被印為
   「該格不可用於跨條件比較」（`optimize.py:587-591`）。
3. 加性 baseline 的 L∞ 在同 LPIPS 下小 28.2 倍，該 hinge 對它幾乎不啟動。
   於是同一組預設對非加性與加性施加的實際約束不同，「匹配失真的比較」不成立。

**確信度**：高（L∞ 的實測值取自 `DESIGN` §3.2 與 `LOGIC_CHECK` A3，非本次實跑）。
目前沒有可用的驅動腳本設定該值（見 §3 的 `run_defense.py` 問題），故此預設會直接生效。

---

### [重要] `MONITOR_TOL["edit_shift"]` 沿用 SD v1.4／512²／site P 的校準值，且沒有 context 檢查

**位置**：`src/defense/optimize.py:191-232`

**現況**

```python
MONITOR_TOL = {
    "edit_shift": 1e-4,     # E23 實測平均 5.4e-4
    "attn_div": 1e-5,       # 2026-08-04 實測平均 1.6e-4
}

def resolve_stop_tol(cfg_tol, monitor_key) -> float:
    if cfg_tol is not None:
        return cfg_tol
    if monitor_key not in MONITOR_TOL:
        raise KeyError(...)
    return MONITOR_TOL[monitor_key]
```

**應為**：`resolve_lr` 對學習率的處置是「只有 `Calibration.get()` 一個入口，context
不完全相等即拋出」（`optimize.py:235-264`、`calibration.py:100-110`），理由是「一個為
某個對象校準的值被沿用到另一個對象上，而且沒有症狀」。停止門檻是同一種量：
`MONITOR_TOL` 自己的註解說明兩個監看量的動態範圍相差 39 倍，故不可共用；
`src/models/attention.py:462-464` 更明講「`optimize.py` 中 `attn_div` 的平台停止門檻
（1e-5，於 SD v1.4 上實測）都綁在舊層數上，**必須在段 0 重新量測**」。
但 `resolve_stop_tol` 沒有任何 context 參數，也不接受 `Calibration`，模型由 SD v1.4
換成 SDXL、解析度由 512² 換成 1024²、site 由加性像素換成位移場，它一律回傳同一個
1e-4。這是 A2 的形狀被完整保留在另一個量上。

**後果**：`edit_shift` 是 LPIPS 距離，其每步改善量隨模型、解析度、學習率、
site 而變。若 SDXL 1024² 上位移場的平均每步改善低於 1e-4，全部 N1／N2 的格子會在
`min_steps=25` 之後立刻判定收斂並回報「已收斂」；若高出兩個量級，停止準則等於沒開。
兩種情形在外部都沒有症狀。`shared_mass` 被刻意留為拋出（`optimize.py:206-210`）是對的，
但 `edit_shift` 的 1e-4 同樣未針對本輪校準，卻是靜默放行的。

**確信度**：高。

---

### [重要] `align()` 實作的階段一與 `DESIGN` §4 的「原樣保留」不符，且該偏離未被記錄在它自稱的位置

**位置**：`src/defense/optimize.py:123-137`、`optimize.py:606-700`；
對照 `docs/DESIGN_2026-08-05.md:150`

**現況**

```python
# optimize.py:126-130
    # 借鑑 APA（arXiv 2506.01511）「先立保真、再訓攻擊」的順序。APA 的階段一
    # 是噪聲空間的 ‖ε − ε̂‖²；此處是影像空間的 L_fid(G(x;φ), x)，因為我們的 G
    # 是固定長度的可微 DDIM 鏈，能端對端反傳到輸出影像本身，不需要噪聲空間的
    # 代理量。**這是一個有記錄的偏離**，見 `DESIGN` §4 的移植表。
```

而 `DESIGN` §4 的移植表該列寫的是：

```
| 一：視覺一致性 | `R_s(Δθ) = E_{t,ε}[−‖ε − ε_{θ+Δθ}(z_t,t,c)‖²]`，LoRA | **原樣保留**，LoRA 掛在 SDXL UNet 的 cross-attention 線性層 |
```

`src/residual/site_apa.py:9` 也重複了「**原樣保留**」。全 repo 搜尋不到任何噪聲空間
`‖ε − ε̂‖²` 的階段一實作。

**應為**：兩者必居其一。若採影像空間的 `L_fid`，`DESIGN` §4 與 `site_apa.py` 的移植表
必須改為「已替換」並附 before/after（本專案的硬性要求）；若要「原樣保留」，
則需另實作噪聲空間的 reward。目前的狀態是**程式做的是 A、文件宣稱的是 B，而程式的註解
指向文件裡並不存在的那條紀錄**。

**後果**：論文若照 `DESIGN` §4 陳述「階段一原樣保留 APA 的 `R_s`」，該陳述與實作不符。
此外兩者的優化目標實質不同：APA 的 `R_s` 是在噪聲空間對**任意 t 與 ε** 取期望，
本實作是在固定 `k_inv` 步 DDIM 鏈的輸出影像上取保真度，兩者對 LoRA 的塑形方向並不等價，
N3 的「視覺一致性」這一半不能宣稱與原論文相同。

**確信度**：確定（文件與程式對讀可得）。

---

### [重要] `config_hash` 的必填鍵未涵蓋參數化容量，A7 的情形不會被雜湊分開

**位置**：`src/utils/cellid.py:35-51`

**現況**

```python
REQUIRED_KEYS = (
    "spec_version", "model", "resolution", "guidance", "steps", "strength",
    "gpu", "precision", "condition", "loss_params", "lr", "tau", "purify",
    "seed", "image_id",
)
```

**應為**：`LOGIC_CHECK` A7 的原文是「**同一 site 的不同設定不可合併統計**（控制點 32 與
128 平均會抹掉效應）」。「控制點」即 `WarpResidual.grid_size`。必填鍵中有代表損失的
`loss_params`，卻**沒有代表參數化的欄位**——`grid_size`、`max_disp`、`resample`、
LoRA 的 `rank`／`alpha`／`blocks`、`LatentResidual` 的 `max_rank`／`const_rank`／`scale`
全部不在必填之列。同樣不在的還有會改變結果的執行參數：`k_inv`、`t_max`、
`exact_inversion`、`n_edit`、`n_eot`、`purify_mode`、`stages`（含 `max_steps`）、
`stop_*`、`attn_timesteps`、`align_*`。應比照 `loss_params` 增設
`module_params` 與 `optim_params` 兩個必填鍵。

**後果**：`check_complete` 只檢查「必填鍵有沒有出現」，額外鍵雖然會進雜湊，但沒有任何
機制強迫呼叫端把容量參數放進去。同一個 `condition="N1"` 在 `grid_size=32` 與
`grid_size=128` 下會算出**相同的 `config_hash`**，於是 `ProgressWriter.is_done`
（`progress.py:214-225`）判定第二組為「已完成」並沿用第一組的產物——這正是 A7 與 A12
兩條缺陷同時重演，且完全沒有症狀。

**確信度**：高（機制確定；實際是否踩到取決於尚未存在的呼叫端，見 §3）。

---

### [重要] `solve_k` 迭代用盡後沒有容差檢查，會靜默回傳未落在 τ 上的解

**位置**：`src/metrics/ray_scale.py:58-74`

**現況**

```python
    lo, hi = 0.0, 1.0
    while lpips_fn(build(hi)) < target:
        hi *= 2.0
        if hi > k_max:
            raise ValueError(...)
    k = hi
    for _ in range(iters):
        k = 0.5 * (lo + hi)
        x = build(k)
        got = lpips_fn(x)
        if abs(got - target) < tol:
            return x, got, k
        lo, hi = (k, hi) if got < target else (lo, k)
    x = build(k)
    return x, lpips_fn(x), k          # ← 無容差檢查
```

**應為**：模組 docstring 自己寫「**到不了就拋出**……不靜默取最接近的值——『達不到 0.5』
與『達到了 0.5』在下游完全不同」。該規則只實作在**上界擴張**那一段；二分迴圈用盡
`iters` 後的回傳沒有做同一件事。此外二分的前提是 `lpips(build(0)) < target`，
而生成路徑（N3）的 `build(0)` 已有 VAE 來回的 LPIPS 0.1434 底線，
對任何 `target < 0.1434` 該前提不成立，迴圈會把 k 壓向 0 卻永遠達不到 target，
最後從第 73 行靜默回傳一個 LPIPS ≈ 0.1434 的影像。

**後果**：這是 A3（匹配失真已被證偽四次）的直接復發路徑。本輪把「匹配失真」的責任
整個交給射線縮放（`DESIGN` §3.2），若該函式可以回傳偏離 τ 的結果而不出聲，
主表上標為「τ = 0.20」的格子實際失真可能是別的值，而下游沒有任何欄位看得出來。
建議在第 73 行前加上與擴張段同樣的 `abs(got - target) < tol` 檢查，不符即拋出。

**確信度**：確定（程式路徑可直接讀出）。目前無呼叫端，故尚未產生錯誤資料。

---

### [重要] N1 的注意力目標取自空 prompt 的無條件分支，與攻擊時實際定位編輯的分支不同

**位置**：`src/defense/optimize.py:742-743`、`optimize.py:898-918`（特別是第 912 行）

**現況**

```python
# optimize.py:743
    emb_cond = sd.encode_text("").detach()
...
# optimize.py:910-913
            zt = abar[t].sqrt() * z_def + (1 - abar[t]).sqrt() * n
            with rec:
                sd._eps(zt, t, emb_cond)   # 見 docstring：此處不開 checkpoint
```

**應為**：`DESIGN` §2.1 把 N1 的形式對應到 PromptFlare 的 cross-attention decoy，
其 `L_CA` 是對**攻擊 prompt 之下**的 `M^BOS` 取的——BOS 質量之所以是有意義的 decoy，
是因為它與同一序列中的內容 token **在同一個 softmax 裡競爭質量**。本實作餵的是空 prompt，
其 77 格是 `[BOS][EOS][PAD]×75`，序列中根本沒有內容 token 可以競爭。
最佳化「在無內容 token 的序列上把質量推向 BOS」與「在有內容 token 的序列上把質量搶離
內容 token」是兩件事，前者未必蘊含後者。

**後果**：若不成立，N1 訓練出來的 φ 在攻擊方以 guidance 7.5 + 真實 prompt 走條件分支時
不產生預期的 decoy 效果，而訓練期的 `shared_mass` 曲線會漂亮地上升——量到的不是我們
以為的東西。這一項無法由程式邏輯本身判定，必須以實測驗證：在同一個 φ 上，
分別記錄空 prompt 與含內容 token 的 prompt 兩種嵌入下的 `shared_mass`（或內容 token 質量），
確認前者上升時後者確實下降。建議排進段 0。

**確信度**：推測（機制上的疑慮，未實測）。

---

## 3. 次要

### [次要] `result.x_def` 比最終 φ 落後一個優化步

**位置**：`src/defense/optimize.py:846`、`optimize.py:917`、`optimize.py:536-548`

`step_fn` 內以 `result.x_def = x_def.detach().clone()` 記錄，而 `opt.step()` 在
`step_fn` 回傳之後才執行。故留存的防禦圖對應 φ_k，而模塊最終持有的是 φ_{k+1}。
評測若用 `result.x_def`、參數快照若存最終 `state_dict`，兩者不是同一個 φ。
影響量級小（一個 Adam 步），但會讓「重跑同一個 φ 得到同一張圖」這條可驗證性質不成立。
**確信度**：確定。

### [次要] `is_done` 在 `artifacts` 為空清單時退化成只看狀態與雜湊

**位置**：`src/utils/progress.py:225`

`all(... for a in cell.get("artifacts", []))` 對空清單恆為 True。`finish()` 的
`artifacts` 是可選參數，未給時就是空清單。docstring 宣稱的三條件（狀態、雜湊、產物存在）
在該情形下只剩兩條，「產物被清掉」會被誤判為完成。建議 `finish()` 在
`artifacts` 為空時拒絕，或 `is_done` 對空清單回傳 False。
**確信度**：確定。

### [次要] 唯一的驅動腳本 `scripts/run_defense.py` 無法 import，本輪沒有可執行的入口

**位置**：`scripts/run_defense.py:31-32`、`551-585`

實跑結果：

```
ImportError: cannot import name 'optimize_crossattn' from 'src.defense.optimize'
```

該檔仍以 `steps=`、`lr=`、`align_lr=`、`attn_mode=`、`color_max_dev=`、`prompt_edit=`
建構 `OptimConfig`，而這些欄位都已在本輪移除或改名（`lr` 改為 `stages[].lr_key`）。
症狀是「拋出」而非「靜默跑錯」，故不升級為重要；但這代表 `optimize()` 的新介面
（`stages`、`calib`、`calib_context`、`y_target`）**沒有任何端到端呼叫端**，
§2 列出的四項預設值問題（`beta_linf`、`stop_require_feasible`、`MONITOR_TOL`、
`config_hash` 的必填鍵）目前都還沒有被實際設定覆蓋過。
**確信度**：確定（實跑）。

### [次要] `APA_STAGE1_LR` 等常數未被使用，是繞過 `resolve_lr` 的潛在入口

**位置**：`src/residual/site_apa.py:53-55`

`APA_STAGE1_LR = 1e-4`、`APA_STAGE1_STEPS = 200`、`APA_NOISE_OFFSET = 0.1` 三者
在 repo 中沒有任何讀取端。學習率的硬規則是「只能由 `Calibration.get()` 取得，
沒有預設值、沒有回退」（A2）；把一個具名的學習率常數放在模組頂層，等於留了一條
「直接用它就好」的路。建議刪除或改為僅出現在文件中的原始碼查證紀錄。
**確信度**：確定。

### [次要] `CompositeResidual._any_pixel()` 對巢狀複合模塊誤判

**位置**：`src/residual/composite.py:101-102`

```python
return any(type(m).pixel_residual is not ResidualModule.pixel_residual
           for m in self.members)
```

成員若本身是 `CompositeResidual`，`type(m).pixel_residual` 恆不等於基底類別的實作，
故即使其全部子成員都不是像素側，也會回報 True。`generator.prepare()` 正是用
「有沒有 `pixel_residual`」決定走哪條路徑（`generator.py:69`），誤判會讓巢狀複合模塊
在停用時被送進像素側路徑。目前沒有巢狀 APA 的用例，故列為次要。
**確信度**：確定。

### [次要] `fidelity_term` 對帶梯度的張量呼叫 `float()`，產生 UserWarning

**位置**：`src/defense/objective.py:427`

```python
"fid_linf_total": float((xd - xr).abs().max()),
```

同一函式的 `fid_acut`／`fid_pen_acut`／`fid_chroma`／`fid_pen_chroma` 都顯式
`.detach()`，此行沒有。數值正確，但測試輸出有
`UserWarning: Converting a tensor with requires_grad=True to a scalar`。
**確信度**：確定（實跑可見）。

### [次要] 測試名稱與斷言不一致

**位置**：`tests/test_objective.py:721-733`

函式名為 `test_色度偏壓約束預設開啟且門檻為零點六`，docstring 亦寫「0.6 由人眼判讀」，
但斷言的是 `c.tau_chroma == 0.8`，且 `objective.py:176-179` 說明取 0.8 而非 1.0 的理由。
測試本身正確，名稱與 docstring 未同步。
**確信度**：確定。

---

## 4. 測試可信度

### 4.1 假通過：`test_φ為零時eps_hook不改變預測噪聲`

**位置**：`tests/test_composite.py:49-54`、`78-84`

```python
def _apa_like():
    ...
    lat = LatentResidual(steps=4, channels=4, size=8, max_rank=2, const_rank=2,
                         init_std=0.0, seed=0)          # ← 與 build_apa 同一個缺陷

def test_φ為零時eps_hook不改變預測噪聲():
    ...
    assert torch.equal(out, eps), "初值為零時必須逐位元等同"
```

該斷言在「參數化正常（U 高斯、V 零）」與「參數化已死（U、V 皆零）」兩種情形下**同樣通過**，
故它無法區分兩者。`test_composite.py` 全檔沒有任何測試檢查階段二參數在 backward 之後
梯度非零。這正是 §1 的致命問題能夠通過 518 個測試的原因。

**建議補上的斷言**（本報告不修改檔案，僅列出應驗的性質）：

1. `stage2` 的參數在一次 `loss.backward()` 之後 `grad.abs().max() > 0`。
2. `build_apa(...)` 本身要有測試——目前它**沒有任何測試**。
3. 階段一結束後跑階段二，`stage1` 的參數張量逐位元不變（`only_trainable` 的實際效果，
   目前只有「requires_grad 被還原」是隱含被測的，參數不變沒有被斷言）。

### 4.2 `run_stages` 的零梯度防護不涵蓋「梯度存在但恆為零」

**位置**：`src/defense/optimize.py:540-545`

```python
grads = [p.grad for p in params if p.grad is not None]
if not grads:
    raise RuntimeError("...φ 沒有進入計算圖...")
```

該檢查只擋「φ 沒進計算圖」，不擋「φ 進了計算圖但梯度恆為零」。後者正是 A1 與
§1 兩個致命失效的共同形狀。`log["grad_norm"]` 有記錄，但沒有任何自動判定。
建議在第一步斷言 `grad_norm > 0`，理由與 `tests/test_zero_gradient.py` 全檔相同。

### 4.3 通過項的抽查

以下項目查核後**未發現**假通過：

- `tests/test_zero_gradient.py`：兩個 targeted 形式在 φ=0 的梯度非零有實際 backward
  驗證，且 `test_targeted_attn_的梯度指向shared_token` 額外驗了方向（shared 位置為負、
  其餘為零），不只是「有值就好」。被淘汰的 `untargeted` 以手工算式保存，並排除了
  「hinge 飽和」這個競爭解釋（`test_已淘汰的untargeted_的零梯度來自距離而非hinge飽和`）。
- `tests/test_plateau_stop.py`：`test_仍在上升時不停` 用的是 E23 的實測上升率 5.4e-4，
  `test_用edit_shift的門檻會讓注意力路徑在半途停下` 讀真實 run 的 60 步歷史並斷言
  `first_stop(1e-4) == 30`、`first_stop(1e-5) is None`。這兩條是有鑑別力的。
- `pytest.approx` 的使用：抽查 `test_objective.py:700`（`abs=1e-6`）、
  `:667`（`abs=1e-9`）、`test_zero_gradient.py:172`（`abs=1e-6`）、`:240`
  （比值 == 10.0），容差都與被測量的量級相稱，未見寬到失去意義者。
- 全域 RNG 相依：抽查的隨機構造全部以 `torch.Generator().manual_seed(...)` 顯式建立
  （`test_zero_gradient._pair`／`_maps`、`test_objective` 的 20260728 種子）。
  全套執行（518 passed）與本次針對 `site_latent` 的單獨執行結果一致。
- mock：`tests/test_composite.py` 用 `_FakeUNet` 取代 SD 是為了避開權重載入，
  被驗的複合邏輯（能力聚合、參數分組、hook 掛載與卸除）都走真實程式路徑，
  不屬於「mock 掉本該驗的路徑」。

---

## 5. 13 條既知缺陷的逐條檢查結果

| # | 缺陷 | 判定 | 依據 |
|---|---|---|---|
| **A1** | 起點零梯度 | **重蹈（換一種成因）** | 損失層面已處置且有測試（`test_zero_gradient.py` 全檔）：`targeted_output` 與 `targeted_attn` 在 φ=0 的梯度均經實際 backward 驗證非零，`untargeted` 已從 `DEFENSE_MODES` 移除並在 `__post_init__` 具名拋出。但 **N3 階段二在參數初始化層面重現了同一個失效**——`build_apa` 的 `init_std=0.0` 使梯度精確為零（§1，已實測）。`run_stages` 的防護只擋 `grad is None`，擋不住恆零梯度 |
| **A2** | 學習率跨位置沿用 | **已避開（學習率），有疑慮（停止門檻）** | `OptimConfig` 已無 `lr` 欄位；`resolve_lr` 是唯一入口，`calib is None`、鍵未校準、context 不完全相等、值非正四種情形都拋出；`align()` 同樣走 `resolve_lr`。全 repo 只有 `optimize.py:531` 與 `:648` 兩處 `torch.optim.Adam(params, lr=lr)`，`lr` 都來自 `resolve_lr`。**但同一種缺陷在 `resolve_stop_tol` 上完整保留**（§2 第二項）：停止門檻沒有 context 檢查，SD v1.4 的 1e-4 會靜默沿用到 SDXL。另 `site_apa.py:53` 留有未使用的 `APA_STAGE1_LR = 1e-4` 常數 |
| **A3** | 匹配失真四次被證偽 | **有疑慮** | 射線縮放的架構正確（`solve_k` 把 `build` 外提，由呼叫端負責「放大 k 倍」對該參數化的正確定義），但二分迴圈用盡 `iters` 後**沒有容差檢查**，會靜默回傳偏離 τ 的解；生成路徑 `build(0)` 的 LPIPS 底線 0.1434 使該情形在 τ < 0.1434 時必然發生（§2 第五項）。此外 `beta_linf=100.0` 的預設使兩類方法被不同的約束綁住（§2 第一項），這本身就會破壞匹配失真的前提 |
| **A4** | n=1 時 \|mean\| > sd 恆成立 | **不適用（本次範圍外）** | 統計判定不在 `src/defense/`、`src/residual/`。`DESIGN` §7 定 n=5、配對 15，`LOGIC_CHECK` C3 已記錄統計力未估 |
| **A5** | `net_lpips` / `edit_shift` 不是成功判定 | **已避開** | `DEFENSE_MONITOR` 的兩條對應方向都正確：`targeted_output` → `edit_shift`（`objective.py:466-473` 在 `no_grad` 下算 `d(y_def, y_orig)`，隨進展**上升**）；`targeted_attn` → `shared_mass`（`objective.py:483` 記為 `1 − l_def`，而 `l_def = 1 − shared_mass`，隨進展**上升**）。兩者都不是隨進展下降的 `L_def`。`plateau_stop` 對缺鍵與 NaN 都拋出而非略過（`optimize.py:434-448`），`edit_shift` 在注意力路徑記為 NaN 以免被誤用。判定用的主判準（CLIP／SigLIP）不在本模組 |
| **A6** | guidance 必須為 7.5 | **已避開** | `OptimConfig.guidance_scale = 7.5`（`optimize.py:154`）；`cellid.REQUIRED_KEYS` 含 `guidance`，改動會使全部格點雜湊改變。訓練期因 prompt-free 而條件與無條件分支相同，CFG 在數值上恆等，該欄位對訓練無效果但對評測有，設計上已註明 |
| **A7** | 不同設定不可合併統計 | **有疑慮** | `config_hash` 的機制正確（正規化、None 不進雜湊、必填檢查、`repr` 往返表示），但 `REQUIRED_KEYS` **沒有涵蓋參數化容量**——A7 原文點名的「控制點 32 與 128」（即 `grid_size`）不在必填之列，LoRA `rank`、`max_rank`、`k_inv`、`n_edit`、`purify_mode`、`stages` 的 `max_steps` 亦然（§2 第四項）。同一 `condition` 的兩種容量會算出同一個雜湊，`is_done` 會沿用舊結果 |
| **A8** | 已排除的保真度量不可作為約束 | **已避開** | `fidelity_term` 的 `total` 只含 L∞／LPIPS／鈍化／色度／PSNR 五項，全部為 `clamp(·, min=0)` 乘非負係數。SSIM 只出現在 `parts["fid_ssim"]`，由 `piq.ssim` 算出後即取 `float()`，不進 `total`；`tests/test_objective.py:84` 專測此事。NLPD／VIF／GMSD／HaarPSI／ΔE 在 `objective.py` 中完全不存在 |
| **A9** | 每個條件都要有同失真隨機對照 | **已避開（本模組內）** | `src/metrics/ray_scale.py:92-112` 的 `gaussian_control` 已實作，且其 docstring 明載「這個條件是必要的，不是加分項」與 E2 的假陽性紀錄。條件 R 進入 9 個訓練條件（`DESIGN` §3.1）。實際是否每格都配了 R，取決於尚不存在的驅動腳本 |
| **A10** | SD 3.x 起無 cross-attention | **不適用（已規避）** | 選定 SDXL；`CrossAttentionRecorder` 以 `cross_attention_layer_count` 由 config 推導層數並在掃描時比對，不符即拋出（`attention.py:206-217`），不接受「安靜地少記幾層」 |
| **A11** | 生成路徑的 VAE 重建誤差下限 | **已避開，但有一個未處理的下游** | `site_apa.py:29-36` 與 `progress.py` 的 `skipped` 五態都已就緒。**但** `solve_k` 對「target 低於該下限」沒有拋出路徑（§2 第五項），若呼叫端忘了先判 `skipped`，得到的是一個靜默偏離 τ 的結果而不是例外 |
| **A12** | 改動執行中的腳本不影響該次執行 | **已避開** | `spec_version` 在 `REQUIRED_KEYS` 之首；`is_done` 同時檢查狀態、雜湊、產物存在（`progress.py:214-225`）。唯一的洞是 `artifacts` 為空清單時第三項退化（§3 第二項） |
| **A13** | `.gitignore` 的 `runs/` 區塊靜默漏檔 | **已避開** | `Calibration.save` 對非 `.json` 後綴直接拋出（`calibration.py:66-72`）並在 docstring 記錄實測命中的 `.gitignore:16:runs/**`；`progress.py` 的 `PROGRESS_NAME`／`CELLS_DIR` 都用 `.json`。`tests/test_gitignore_paths.py` 為迴歸測試 |

**統計**：已避開 8 條、有疑慮 3 條（A2 的停止門檻半邊、A3、A7）、重蹈 1 條（A1，經由參數初始化）、不適用 1 條（A4；A10、A11 歸入已避開但各附一項下游備註）。

---

## 6. 建議的處理順序

1. `src/residual/site_apa.py:99` 的 `init_std=0.0` 改為正值（例如沿用 `LowRankResidual`
   的預設 0.02），並補上「stage2 參數梯度非零」與「stage1 參數在 stage2 中逐位元不變」
   兩條測試。在此之前 N3 的任何結果都不可採用。
2. 在 `run_stages` 的第一步斷言 `grad_norm > 0`，讓 A1 這一類失效有具名攔截點。
3. 決定 `beta_linf` 在本輪的值（既有 run 全部用 0.0），並確認 `stop_require_feasible`
   與 `active_constraint_keys` 的組合對位移場條件可達。
4. `resolve_stop_tol` 改為與 `resolve_lr` 同構：向 `Calibration` 索取並檢查 context，
   或把 `edit_shift` 的門檻一併移出 `MONITOR_TOL`，強迫段 0 重新量測。
5. `cellid.REQUIRED_KEYS` 增設 `module_params` 與 `optim_params`。
6. `solve_k` 在迭代用盡後補容差檢查並拋出。
7. 統一 `DESIGN` §4 與 `site_apa.py` 移植表對階段一的陳述，附 before/after。
8. 段 0 加一項驗證：同一個 φ 下，空 prompt 的 `shared_mass` 上升時，含內容 token 的
   prompt 下內容 token 的注意力質量確實下降。
