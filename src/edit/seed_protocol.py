"""Seed 協定 — SPEC.md §2.3、STRUCTURE.md §2.3。

先在原圖上搜尋能產生合理編輯結果的 seed，再以同一 seed 清單編輯受保護影像，
確保兩者編輯條件完全相同（實驗公平性之必要條件）。DAYN 以 20 個 seed 平均。

「合理」的判準需與指導者確認；預設以編輯結果與 prompt 的 CLIP score 高於門檻
（config["edit"]["seed_clip_threshold"]）為準。門檻為 null 時不篩選，
直接回傳自 base seed 起之連續 seed（決定性）。
"""

import torch

from src.edit import edit_image


def search_valid_seeds(
    image: torch.Tensor,
    prompt: str,
    n_seeds: int = 20,
    config: dict = None,
    *,
    sd=None,
    clip_scorer=None,
) -> list[int]:
    """回傳 n_seeds 個通過判準的 seed，將同時用於原圖與受保護影像之編輯。"""
    base = (config or {}).get("runtime", {}).get("seed", 0)
    threshold = (config or {}).get("edit", {}).get("seed_clip_threshold")

    if threshold is None:
        return list(range(base, base + n_seeds))

    if sd is None or clip_scorer is None:
        raise ValueError("啟用 CLIP 門檻篩選需提供 sd 與 clip_scorer")

    valid, candidate, limit = [], base, base + 100 * n_seeds
    while len(valid) < n_seeds and candidate < limit:
        edited = edit_image(image, prompt, candidate, "sdedit", config=config, sd=sd)
        if clip_scorer.score(edited, prompt) >= threshold:
            valid.append(candidate)
        candidate += 1
    if len(valid) < n_seeds:
        raise RuntimeError(
            f"於 {limit - base} 個候選內僅找到 {len(valid)}/{n_seeds} 個合格 seed，"
            f"請檢視門檻 {threshold} 是否過嚴"
        )
    return valid
