"""保護方法：加性（PhotoGuard）與非加性（AdvDiff-based、APA-based、Hybrid）。

build_protection() 為 stage 腳本統一入口，方法鍵與 STRUCTURE.md §4.3 方法矩陣對應：
- pg_enc / pg_diff：configs/additive.yaml 之 photoguard 區塊
- advdiff / apa / hybrid：configs/nonadditive.yaml 之對應區塊
- apa_sg / apa_gc：同 apa 但強制指定 variant（stage0 兩變體獨立校準用）
- hybrid 之完整 cfg = apa 區塊 + hybrid 區塊之覆蓋鍵（如校準後 eps_a）
  + advdiff 之 s、a（見 nonadditive.yaml hybrid 註）
"""

METHOD_KEYS = ["pg_enc", "pg_diff", "advdiff", "apa", "hybrid"]

_HYBRID_META_KEYS = ("base", "injection")  # hybrid 區塊中非參數之描述鍵


def build_protection(key: str, sd, additive_cfg: dict, nonadditive_cfg: dict):
    from src.protect.advdiff_based import AdvDiffProtection
    from src.protect.apa_based import APAProtection
    from src.protect.hybrid import HybridProtection
    from src.protect.photoguard import PhotoGuardDiffusion, PhotoGuardEncoder

    if key == "pg_enc":
        return PhotoGuardEncoder(sd, additive_cfg["photoguard"])
    if key == "pg_diff":
        return PhotoGuardDiffusion(sd, additive_cfg["photoguard"])
    if key == "advdiff":
        return AdvDiffProtection(sd, nonadditive_cfg["advdiff"])
    if key == "apa":
        return APAProtection(sd, nonadditive_cfg["apa"])
    if key in ("apa_sg", "apa_gc"):
        return APAProtection(sd, {**nonadditive_cfg["apa"], "variant": key[-2:]})
    if key == "hybrid":
        adv = nonadditive_cfg["advdiff"]
        overrides = {k: v for k, v in nonadditive_cfg.get("hybrid", {}).items()
                     if k not in _HYBRID_META_KEYS}
        cfg = {**nonadditive_cfg["apa"], **overrides,
               "variant": "sg", "s": adv["s"], "a": adv["a"]}
        return HybridProtection(sd, cfg)
    raise ValueError(f"未知保護方法: {key}")
