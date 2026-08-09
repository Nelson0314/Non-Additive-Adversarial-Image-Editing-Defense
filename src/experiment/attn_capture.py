"""編輯路徑上的 cross-attention 擷取與落盤 —— `reference/CODE_CONTRACTS.md` §4.2。

**這是主判準的一部分，不是附屬產物。** 對比論文中 AdvPaint（self- 與
cross-attention 擾亂）、PromptFlare（cross-attention decoy）與本專案的 N1
都以 attention 為著力點；沒有 attention map 就無法說明「防禦是否真的讓
那些層失效」，也無法回答「淨化是不是把 attention 的擾亂洗掉了」。

## 取樣規則（`CODE` §4.2，非本模組自訂）

| 項目 | 規則 |
|---|---|
| 擷取方式 | `CrossAttentionRecorder` 的 forward pre-hook，**不換 processor** |
| 涵蓋範圍 | 全部 `attn2` 層（SDXL 70、SD v1.5 16），存圖時依解析度分組 |
| 時間軸 | 每 `STEP_EVERY` 步一次，**固定含第 0 步與最後一步** |
| 兩側 | 防禦側與 φ=0 對照側都要，否則無法相減 |
| 淨化後 | 每個淨化算子的編輯路徑也要 |
| 聚合圖 | 另存 `aggregate_token_attention` 的跨層聚合圖 |
| 數值 | `attn_stats.csv` 存逐層逐步的統計量，使圖可重繪、結論可重算 |

**體積控制**：逐層原圖只在 `attn_full` 的那一組完整存（主表所在的
τ = 0.20、seed 0），其餘格點只存聚合圖與 `attn_stats.csv`。
理由是體積，**不是因為其餘格點不重要**——`attn_stats.csv` 每一格都有，
數值結論不受影響。該旗標寫進 `meta.json`。

## 只取條件分支

`_eps_cfg` 對無條件與條件各做一次前向，順序是先無條件後條件，故 recorder
在一步之內會收到 `2 × n_layers` 張圖。**只保留條件分支那一半**：stock SDXL
base 的 `force_zeros_for_empty_prompt=true`，無條件分支的嵌入是零張量，
其注意力不承載任何文字綁定，混進來只會把統計量往均勻分佈拉。
`guidance_scale == 1.0` 時只有一次前向，全部都是條件分支。

## 為什麼逐層圖是跨步平均而不是逐步各存

檔名（`CODE` §4.2）是 `tau{τ}_seed{k}_res{R}_layer{L}.png`，沒有步的維度。
70 層 × 11 個取樣步逐一存會是 770 張圖乘上格數，而判讀「哪一層被打掉」
看的是該層在整條軌跡上的平均行為。逐步的數值仍完整留在 `attn_stats.csv`
裡，需要重繪某一步時由該表取得。
"""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.models.attention import (
    CrossAttentionRecorder, aggregate_token_attention, token_span,
)

# `CODE` §4.2：每 5 步存一張，固定含第 0 步與最後一步。
STEP_EVERY = 5


def sampled_steps(num_steps: int, every: int = STEP_EVERY) -> List[int]:
    """要擷取的步索引。固定含第 0 步與最後一步。

    最後一步不可省：x̂₀ 在末段才收斂到最終影像，只看前段會把「注意力在
    早期被打散、後期又長回來」誤讀成防禦成功。
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps 必須為正，收到 {num_steps}")
    idx = set(range(0, num_steps, max(1, every)))
    idx.add(0)
    idx.add(num_steps - 1)
    return sorted(idx)


def _side_of(n_tokens: int) -> int:
    """把 Q 還原成方形邊長。**不作長寬比推測**——猜錯會讓圖被轉置而
    完全沒有症狀（見 `aggregate_token_attention` 的同一條規則）。"""
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise ValueError(
            f"注意力的 Q 維度 {n_tokens} 不是完全平方數，無法還原成方形。"
            "非方形影像須由呼叫端提供形狀"
        )
    return side


class AttnCapture:
    """一次編輯路徑的擷取結果。

    用法：

        cap = AttnCapture(sd, num_steps, span)
        with cap:
            y = sd.sdedit(..., step_hook=cap.step_hook)
        cap.write(out_dir, tag="tau0.2_seed0", full=True)
    """

    def __init__(self, sd, num_steps: int, span: Tuple[int, int],
                 every: int = STEP_EVERY):
        self.sd = sd
        self.span = span
        self.steps = sampled_steps(num_steps, every)
        self._want = set(self.steps)
        self.recorder = CrossAttentionRecorder(sd.unet)
        self.n_layers = self.recorder.n_layers
        # layer_index -> 逐步累加的平均圖（(side, side)），最後除以步數
        self._layer_sum: Dict[int, torch.Tensor] = {}
        self._agg_sum: Optional[torch.Tensor] = None
        self._n_sampled = 0
        self.rows: List[Dict[str, Any]] = []

    def __enter__(self) -> "AttnCapture":
        self.recorder.__enter__()
        self.recorder.enabled = False
        return self

    def __exit__(self, *exc):
        return self.recorder.__exit__(*exc)

    # ---- 逐步 ----

    def step_hook(self, i: int, t, pred_x0) -> None:
        """`sdedit` 每步呼叫兩次：前向之前（`pred_x0` 為 None）與之後。"""
        if pred_x0 is None:
            self.recorder.enabled = i in self._want
            self.recorder.clear()
            return
        if i not in self._want:
            return
        maps = self._cond_branch(self.recorder.maps)
        self._accumulate(i, int(t), maps)
        self.recorder.clear()
        self.recorder.enabled = False

    def _cond_branch(self, maps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """CFG 下取後半（條件分支），見模組 docstring。"""
        n = self.n_layers
        if len(maps) == n:
            return list(maps)
        if len(maps) == 2 * n:
            return list(maps[n:])
        raise RuntimeError(
            f"一步之內記到 {len(maps)} 張注意力圖，既不是 {n}（無 CFG）"
            f"也不是 {2 * n}（CFG 兩次前向）。層數或前向次數與預期不符，"
            "此時分不出哪些是條件分支，統計量不可用"
        )

    def _accumulate(self, step: int, t: int, maps: List[torch.Tensor]) -> None:
        agg = aggregate_token_attention(maps, self.span, reduce="mean")
        self._agg_sum = agg.detach().float().cpu() if self._agg_sum is None \
            else self._agg_sum + agg.detach().float().cpu()
        self._n_sampled += 1

        lo, hi = self.span
        for li, a in enumerate(maps):
            a = a.detach().float()
            content = a[..., lo:hi].sum(dim=-1)          # (B, Q)
            side = _side_of(a.shape[1])
            img = content[0].reshape(side, side).cpu()
            self._layer_sum[li] = img if li not in self._layer_sum \
                else self._layer_sum[li] + img
            # 熵在 token 維度上算：分佈趨近均勻代表沒有任何 token 主導任何
            # 位置，即綁定被瓦解。與質量分開記，兩者會分歧——質量降低但熵
            # 不變表示注意力被整體壓低而非被打散。
            p = a.clamp_min(1e-8)
            ent = (-(p * p.log()).sum(dim=-1)).mean()
            self.rows.append({
                "step": step, "t": t, "layer": li, "side": side,
                "content_mass_mean": float(content.mean()),
                "content_mass_max": float(content.max()),
                "entropy": float(ent),
            })

    # ---- 落盤 ----

    @property
    def n_sampled(self) -> int:
        return self._n_sampled

    def agg_map(self) -> torch.Tensor:
        if self._agg_sum is None:
            raise RuntimeError(
                "沒有任何取樣步被記錄。`step_hook` 沒有接上 `sdedit`，"
                "或 recorder 未被 `with` 啟用"
            )
        return self._agg_sum / self._n_sampled

    def write(self, out_dir: Path, tag: str, full: bool) -> List[Path]:
        """落盤。`full=False` 時只寫聚合圖與 `attn_stats.csv`（體積控制）。"""
        from src.experiment.executors import write_csv
        from src.utils.artifacts import save_heatmap

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [save_heatmap(self.agg_map(), out_dir / f"{tag}_agg.png")]

        if full:
            n = self._n_sampled
            for li, s in sorted(self._layer_sum.items()):
                side = s.shape[-1]
                paths.append(save_heatmap(
                    s / n, out_dir / f"{tag}_res{side}_layer{li:02d}.png"))

        paths.append(write_csv(out_dir / "attn_stats.csv", self.rows))
        return paths


def capture_span(sd, prompt: str) -> Tuple[int, int]:
    """內容 token 的切片。prompt 為空時退回全 77 格。

    空 prompt（本專案的防禦端一律 prompt-free）沒有「內容 token」可言，
    此時取全部 token 的質量——那個量仍有意義（注意力總質量如何分佈），
    只是不再是「分給某個詞的質量」。兩者不可混為一談，故此處明確分派
    而不是讓 `token_span` 對空字串回傳一個看起來正常的區間。
    """
    if not prompt:
        return (0, sd.tokenizer.model_max_length)
    return token_span(sd.tokenizer, prompt)
