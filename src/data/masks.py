"""由 cross-attention 取一個詞所對應的區域，即 Lo et al. 式 (3)(4)。

    Att(x, c_a) = Σ_l upsample(A_l(x, c_a))      式 (3)
    M = I(Att / max(Att) > tau)                  式 (4)，以峰值正規化

兩者都實作於 `src/models/attention.py`；本模組只負責把它套到一張影像上
並做後處理。

**這不是 inpainting 遮罩的來源。**
──────────────────────────────────────────────────────────────────────
2026-08-09 起，攻擊方的遮罩改為**人工繪製**，逐影像一張 PNG 存在
`data/lo_aligned/masks/`，工具是 `scripts/draw_masks.py`（DEC-010）。
文獻的作法即如此：PIE-Bench 附標註遮罩、PhotoGuard 與 AdvPaint 的
inpainting 實驗用人工遮罩、Lo et al. Figure 3 那張也是手畫的；真實的
inpainting 軟體本來就是讓使用者自己框。

before：遮罩由 `content_mask` 依 c_a 的注意力產生，於是它與式 (4) 的 M
完全重疊（DEF-011）——M 落在會被整片覆寫的區域、防禦擾動一步都活不過，
而 `--warp-mask-gate` 又把擾動推到遮罩外，使損失看的地方與擾動所在的地方
不相交。ip1／ip2／ip3 三批都是這個配置且**沒有任何症狀**。

期間曾改為由 `edit_region`（`prompts[1]` 裡要新增的物件詞）的注意力產生並
扣掉 c_a 的保護帶，但那條路自帶一個問題：該物件在原圖裡還不存在，模型會
把注意力放在畫面中最像它的地方——"cow" 在只有馬的照片上多半就落在馬身上。
人工繪製沒有這個問題，且與文獻一致。該作法已移除。

`content_mask` 保留的用途
──────────────────────────────────────────────────────────────────────
式 (4) 的 M 本身仍由 c_a 的注意力取得（真正進損失的那張在
`src/defense/optimize.py`）。本模組的 `content_mask` 是它套在單張影像上的
形式，供診斷與視覺化用——例如要看「模型認為 c_a 在哪裡」與人工遮罩有沒有
打架。人工遮罩與 M 不相交這條不變量由
`src.models.attention.assert_masks_disjoint` 在 `optimize.py` 斷言。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Sequence

import torch

from src.models.attention import (
    aggregate_token_attention,
    attention_region_mask,
    token_span,
)

MASK_MODES = ("attention", "attention_box", "center_box")

# 人工遮罩的二值化門檻，作用在 8-bit 尺度上。diffusers 的 inpainting 管線
# 本來就會對遮罩取門檻，此處先做掉並記錄，使「遮罩其實是灰的」不會變成一個
# 只在管線深處生效、事後查不到的差異。
MASK_BINARIZE = 128


def load_drawn_mask(path: Path, size: int, device) -> torch.Tensor:
    """讀一張人工繪製的遮罩，回傳 (1,1,size,size) 的 0／1 張量。

    **1 表示要重畫的區域**，與 `SDWrapper.inpaint` 及 diffusers 同一約定。
    來源是 `scripts/draw_masks.py` 存出的 8-bit 灰階 PNG（0／255）。

    三道驗證。它們攔的都是「跑得完但量到的不是防禦」的型態：

    - 檔案不存在 → 拋出並指名工具。**不落回任何自動產生的遮罩**：那會讓
      一批裡有些影像用人工遮罩、有些用模型產的，而表格看不出差別。
    - 遮罩為空 → 攻擊方什麼都不重畫，該影像的每一格都在量一個不存在的攻擊。
    - 遮罩填滿整張 → c_a 不可能落在遮罩外，即 DEF-011 的配置。

    尺寸不符時以 NEAREST 縮放：遮罩是二值的，任何插值都會在邊界造出中間值。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"找不到人工遮罩 {path}。inpainting 的遮罩是攻擊方的設定，"
            "本專案改為人工繪製（DEC-010）——先跑 "
            "`python scripts/draw_masks.py` 把它畫出來。"
            "此處不落回模型自動產生的遮罩：混用兩種來源會讓同一張表上的"
            "各列不可比，而那看不出症狀"
        )
    from PIL import Image

    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    # `bytearray` 而非 `bytes`：後者不可寫，`frombuffer` 會對它發警告。
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    m = (t.reshape(1, 1, size, size) >= MASK_BINARIZE).float().to(device)

    cov = float(m.mean())
    if cov == 0.0:
        raise ValueError(
            f"{path} 的遮罩為空。inpainting 下攻擊方什麼都不會重畫，"
            "該影像的每一格都會在量一個不存在的攻擊"
        )
    if cov == 1.0:
        raise ValueError(
            f"{path} 的遮罩填滿整張影像。c_a 不可能落在遮罩外，"
            "這正是 DEF-011 的配置——防禦擾動一步都活不過"
        )
    return m


# 遮罩目錄裡不是遮罩的 PNG。它們由 `draw_masks.py` 產生，是給人看的衍生物，
# 重畫一次就會變——放進 digest 會讓「重新產一張總覽圖」靜默改掉每一格的
# `config_hash`，續跑時把已完成的格全部判為未完成。
NON_MASK_PNGS = frozenset({"overview.png"})


def mask_files(mask_dir: Path) -> list:
    """遮罩目錄裡**真正是遮罩**的 PNG，排序後回傳。

    `glob("*.png")` 會收到 `overview.png`，故不可直接用。子目錄
    （`_subject/`、`_original/`）不遞迴，本來就不會進來。
    """
    return sorted(p for p in mask_dir.glob("*.png")
                  if p.name not in NON_MASK_PNGS)


def masks_digest(paths: Sequence[Path]) -> str:
    """整組人工遮罩的摘要，供 `config_hash` 使用。

    取整組而非逐影像：`base_config` 是批次層級的，而 `cell_config` 疊上去的
    四個軸裡沒有遮罩。整組摘要會讓**改一張遮罩使全部格的雜湊改變**——那是
    刻意的保守方向。遮罩決定攻擊方能改哪一塊，改了它舊結果就不可沿用；
    寧可多重跑，不可靜默沿用。
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _closing(m: torch.Tensor, k: int = 5) -> torch.Tensor:
    """二值遮罩的形態學閉運算（先膨脹再侵蝕），填掉注意力圖上的小孔。

    以最大池化實作膨脹、對補集最大池化實作侵蝕；`k` 為奇數方形結構元素。
    孔洞會讓攻擊方在物件內部留下不能改的小塊，那不是任何真實攻擊方會畫的
    遮罩，且它造成的斑點在輸出上看起來像防禦生效，是一個假陽性的來源。
    """
    pad = k // 2
    d = torch.nn.functional.max_pool2d(m, k, stride=1, padding=pad)
    e = 1.0 - torch.nn.functional.max_pool2d(1.0 - d, k, stride=1, padding=pad)
    return e


def _bounding_box(m: torch.Tensor) -> torch.Tensor:
    """遮罩的外接矩形。最接近人工畫框的形態。"""
    idx = (m[0, 0] > 0.5).nonzero()
    if idx.numel() == 0:
        raise ValueError("遮罩為空，取不出外接矩形")
    y0, x0 = idx.min(dim=0).values.tolist()
    y1, x1 = idx.max(dim=0).values.tolist()
    out = torch.zeros_like(m)
    out[..., y0:y1 + 1, x0:x1 + 1] = 1.0
    return out


def center_box(x01: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    """置中方框。**只作為對照**，不是主要作法。

    它與影像內容無關，故「防禦有沒有效」會混入「物件有沒有落在框內」這個
    與本研究無關的變因。保留它是為了讓「遮罩的選法影響多大」有一個下界可比。
    """
    h, w = x01.shape[-2:]
    bh, bw = int(h * frac), int(w * frac)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    m = torch.zeros(1, 1, h, w, device=x01.device, dtype=x01.dtype)
    m[..., y0:y0 + bh, x0:x0 + bw] = 1.0
    return m


def _token_attention(
    sd, x01: torch.Tensor, word: str, *, timestep: int, seed: int
) -> torch.Tensor:
    """單一 timestep 上該詞的聚合 cross-attention（式 3），回傳 (1, s, s)。

    抽成獨立函式，使「取某個詞的注意力」只有一條路徑：timestep、seed、
    reduce 與 9 通道的補法各有一種寫法，分岔出第二份的症狀是兩張圖都長得
    像注意力圖卻不可互相比對。
    """
    from src.models.attention import CrossAttentionRecorder

    emb = sd.encode_text(word).detach()
    span = token_span(sd.tokenizer, word)
    if span[1] <= span[0]:
        raise ValueError(
            f"{word!r} 沒有內容 token（tokenizer 只給出 BOS/EOS）"
        )

    with torch.no_grad():
        z = sd.encode_image(x01)
        noise = sd.sample_edit_noise(z, seed=seed)
        abar = sd.alphas_cumprod(x01.device)
        t = torch.tensor(int(timestep), device=x01.device)
        zt = abar[t].sqrt() * z + (1 - abar[t]).sqrt() * noise
        if sd.is_inpainting:
            # 9 通道權重下，只為了取注意力而跑前向時仍須補滿後 5 個通道。
            # 遮罩此時尚未存在（正在產生它），故給全 1 與原圖的 latent：
            # 那等於「整張都要重畫」，對 attn2 的 query 沒有偏好，不會把
            # 一個尚未決定的遮罩偷偷寫進它自己的來源。
            m1 = torch.ones_like(zt[:, :1])
            zt = torch.cat([zt, m1, z], dim=1)
        rec = CrossAttentionRecorder(sd.unet)
        with rec:
            sd._eps(zt, t, emb)
        att = aggregate_token_attention(rec.maps, span, reduce="mean")
        rec.clear()
    return att


def content_mask(
    sd,
    x01: torch.Tensor,
    content: str,
    *,
    mode: str = "attention_box",
    tau: float = 0.5,
    timestep: int = 500,
    seed: int = 0,
    close_k: int = 5,
) -> Dict[str, object]:
    """由模型對 `content` 的 cross-attention 產生遮罩。

    回傳 `{"mask": (1,1,H,W) [0,1], "coverage": float, "mode": str, ...}`。
    **1 表示要重畫的區域**，與 `SDWrapper.inpaint` 及 diffusers 同一約定。

    實際運算，逐步：

    1. 把 `content` 編碼成 prompt（`token_span` 取出它在 77 格中的區間，
       不含 BOS/EOS——padding 與 EOS 仍會分到可觀的注意力質量卻不承載語意）。
    2. 把原圖編碼成 latent，加噪到 `timestep`。取單一 timestep 而非平均：
       注意力圖在中段 timestep 最能反映物件位置，而逐 t 平均會把早期尚未
       成形的分佈混進來。t=500 是 [0,1000] 的中點。
    3. 跑一次 UNet 前向並以 `CrossAttentionRecorder` 記下全部 attn2 層。
    4. `aggregate_token_attention` 逐層取該詞的質量、上採樣後相加（式 3）。
    5. `attention_region_mask` 以峰值正規化後取 `> tau`（式 4）。
    6. 依 `mode` 後處理，再上採樣到影像尺寸。

    注意力圖的邊長由**實際掃到的層**決定，不是 latent 邊長：SDXL 在 1024²
    下 latent 是 128² 而最細的一層沒有 attention，聚合圖是 64²
    （`attention.cross_attention_resolutions`）。故第 6 步的上採樣是必要的。
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mask_mode 只接受 {MASK_MODES}，收到 {mode!r}")
    if mode == "center_box":
        m = center_box(x01)
        return {"mask": m, "coverage": float(m.mean()), "mode": mode,
                "content": content, "tau": None}
    if not content:
        raise ValueError(
            "content 為空：注意力遮罩需要一個要保護的詞（Lo et al. 的 c_a）。"
            "資料集的 content 欄為必填，此處不落回置中方框——那會讓兩種"
            "來源不同的遮罩混在同一張表上而看不出差別"
        )

    att = _token_attention(sd, x01, content, timestep=timestep, seed=seed)

    m = attention_region_mask(att, tau=tau).unsqueeze(1)     # (1,1,s,s)
    if mode == "attention":
        m = _closing(m, close_k)
    elif mode == "attention_box":
        m = _bounding_box(_closing(m, close_k))

    m = torch.nn.functional.interpolate(
        m, size=x01.shape[-2:], mode="nearest").to(x01.dtype)
    return {"mask": m, "coverage": float(m.mean()), "mode": mode,
            "content": content, "tau": tau, "timestep": int(timestep),
            "attn_side": int(att.shape[-1])}
