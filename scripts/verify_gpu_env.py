"""GPU 機器的環境驗證 — `docs/RUNBOOK_2026-08-05.md` §1、`ARCH` §8。

    python scripts/verify_gpu_env.py [--load-sdxl] [--json]

**連上機器後的第一件事。** 驗的是 `ARCH` §8 標為高風險的項目——
在這些地方出錯，後面全部作廢，而且多數錯法沒有症狀。

## 為什麼這些項目要在跑任何實驗之前驗

| 項目 | 錯了會怎樣 |
|---|---|
| torch 與卡的相容性 | V100 需 cu118、RTX 5090（sm_120）需 ≥2.7+cu128。不符會直接失敗，這是唯一會明確報錯的一項 |
| `attn2` 層數 | SDXL 應為 70、SD v1.5 為 16。掃錯層數則注意力損失的正規化整個錯，而輸出仍是一張合理的圖 |
| `force_zeros_for_empty_prompt` | 為真時 CFG 的無條件分支是零張量。取錯則模擬的攻擊方不是 stock 模型 |
| fp16 下的 VAE | SDXL 原生 VAE 在 fp16 會溢位出**全黑圖**。這一項有症狀，但等到跑完才發現就浪費了整批機時 |
| bf16 支援 | V100（sm_70）沒有 bf16。誤用會退化成極慢的模擬路徑 |

不下載權重也能驗前兩項（層數由 config 純算術推導），故預設不載入模型；
`--load-sdxl` 才會實際下載並驗證後三項。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = []


def check(name: str, ok: bool, detail: str = "", fatal: bool = False):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail,
                    "fatal": fatal})
    mark = "OK  " if ok else ("FATAL" if fatal else "WARN")
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def verify_torch() -> dict:
    import torch

    info = {"torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "available": torch.cuda.is_available()}
    if not check("CUDA 可用", info["available"], f"torch {info['torch']}",
                 fatal=True):
        return info

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    info.update(gpu=name, capability=f"sm_{cap[0]}{cap[1]}", vram_gb=round(total, 1))
    check("GPU", True, f"{name}  sm_{cap[0]}{cap[1]}  {total:.1f} GB")

    # sm_120（Blackwell）需要 torch ≥ 2.7 + CUDA 12.8。cu118 會以
    # `no kernel image is available` 失敗——這是唯一會明確報錯的一項，
    # 但先驗出來比在訓練第一步才炸好。
    if cap[0] >= 12:
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        check("Blackwell 需 torch ≥ 2.7", (major, minor) >= (2, 7),
              f"目前 {torch.__version__}", fatal=True)

    # bf16：V100（sm_70）沒有。它決定 VAE 能否走半精度（見 resolve_precision）
    bf16 = torch.cuda.is_bf16_supported()
    info["bf16"] = bf16
    check("bf16 支援", True, "是（VAE 可走半精度）" if bf16
          else "否（fp16，VAE 必須留 fp32）")

    # 實際跑一次矩陣乘法：能建立 context 不代表能執行 kernel
    try:
        a = torch.randn(256, 256, device="cuda")
        torch.cuda.synchronize()
        check("kernel 可執行", bool(float((a @ a).sum()) == float((a @ a).sum())))
    except Exception as e:
        check("kernel 可執行", False, f"{type(e).__name__}: {e}", fatal=True)
    return info


def verify_layer_count():
    """不需權重：層數由 config 的欄位純算術推導。"""
    from src.models.attention import cross_attention_layer_count

    sdxl = {
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D",
                             "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                           "UpBlock2D"],
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "transformer_layers_per_block": [1, 2, 10],
        "block_out_channels": [320, 640, 1280],
        "layers_per_block": 2,
        "cross_attention_dim": 2048,
    }
    got = cross_attention_layer_count(sdxl)
    check("SDXL 的 attn2 層數推導 = 70", got["total"] == 70, str(got))


def verify_sdxl(model_id: str, precision: str):
    """需要權重。驗載入、層數實掃、無條件分支、以及 fp16 的 VAE 黑圖問題。"""
    import torch

    from src.models.sd import SDXLWrapper
    from src.utils.device import resolve_precision

    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[precision]
    backbone, vae_dtype = resolve_precision(dtype)
    check("精度規則", True,
          f"backbone={backbone}  vae={vae_dtype}"
          + ("（VAE 留 fp32 以避免溢位）" if vae_dtype != backbone else ""))

    print(f"  載入 {model_id}（首次會下載約 7 GB）…", flush=True)
    sd = SDXLWrapper(model_id, dtype=dtype)

    n = sum(1 for name, _ in sd.unet.named_modules() if name.endswith("attn2"))
    check("實掃 attn2 層數 = 70", n == 70, f"實得 {n}")

    flag = sd.force_zeros_for_empty_prompt
    check("force_zeros_for_empty_prompt", flag is True,
          f"{flag}（stock SDXL base 應為 True）")

    u = sd.uncond_prompt()
    c = sd.encode_text("a photo of a cat")
    check("無條件分支為零張量", float(u.embeds.abs().max()) == 0.0)
    check("兩支形狀一致", u.embeds.shape == c.embeds.shape,
          f"{tuple(u.embeds.shape)}")
    check("cross_attention_dim = 2048", c.embeds.shape[-1] == 2048,
          str(c.embeds.shape[-1]))

    # VAE 黑圖：SDXL 原生 VAE 在 fp16 下中間激活會超過 65504 而變 inf。
    # 這一項有症狀，但等到跑完才發現就浪費了整批機時。
    with torch.no_grad():
        x = torch.rand(1, 3, 1024, 1024, device=sd.device)
        z = sd.encode_image(x)
        out = sd.decode_latent(z)
    finite = bool(torch.isfinite(out).all())
    spread = float(out.max() - out.min())
    check("VAE 來回未溢位", finite and spread > 0.05,
          f"finite={finite}  值域跨度={spread:.4f}"
          + ("（跨度過小＝疑似黑圖）" if spread <= 0.05 else ""), fatal=True)

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    check("VAE 來回的峰值記憶體", True, f"{peak:.2f} GB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GPU 環境驗證")
    ap.add_argument("--load-sdxl", action="store_true",
                    help="實際載入 SDXL（需下載約 7 GB）")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--precision", default="fp16",
                    choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    info = verify_torch()
    if any(r["fatal"] and not r["ok"] for r in RESULTS):
        print("\n致命項失敗，後續驗證略過。", file=sys.stderr)
        return 2

    verify_layer_count()
    if args.load_sdxl:
        verify_sdxl(args.model, args.precision)
    else:
        print("\n（未加 --load-sdxl，SDXL 相關的三項未驗）")

    failed = [r["name"] for r in RESULTS if not r["ok"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 通過")
    if failed:
        print("未通過：" + "、".join(failed), file=sys.stderr)
    if args.json:
        print(json.dumps({"env": info, "results": RESULTS},
                         ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
