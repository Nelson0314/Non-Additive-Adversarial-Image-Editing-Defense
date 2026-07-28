"""確認 spec §8.1 的八項指標在本環境是否都能實際算出一個數字。

只檢查「能否 import 並在一對隨機影像上跑出有限值」，不檢查數值正確性。
指標一旦缺席就必須在報告中明列，故此檢查的輸出直接對應 spec §8.1 的表格。

執行：source env.sh && python scripts/check_metrics_deps.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def main():
    x = torch.rand(1, 3, 224, 224)
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    ok, missing = [], []

    def probe(name, fn):
        try:
            v = float(fn())
            assert v == v and abs(v) != float("inf"), "非有限值"
            print(f"  {name:10s} OK   {v:.4f}")
            ok.append(name)
        except Exception as e:  # noqa: BLE001 - 這裡就是要知道任何失敗原因
            print(f"  {name:10s} FAIL {type(e).__name__}: {e}")
            missing.append(name)

    print("piq:")
    import piq

    probe("PSNR", lambda: piq.psnr(x, y))
    probe("SSIM", lambda: piq.ssim(x, y))
    probe("LPIPS", lambda: piq.LPIPS()(x, y))
    probe("DISTS", lambda: piq.DISTS()(x, y))
    probe("Linf", lambda: (x - y).abs().max())

    print("pyiqa:")
    try:
        import pyiqa

        probe("NIQE", lambda: pyiqa.create_metric("niqe")(x))
    except Exception as e:  # noqa: BLE001
        print(f"  NIQE       FAIL import: {type(e).__name__}: {e}")
        missing.append("NIQE")

    print("transformers:")
    for name, repo in [
        ("CLIP", "openai/clip-vit-base-patch32"),
        ("SigLIP", "google/siglip-base-patch16-224"),
    ]:
        try:
            from transformers import AutoModel, AutoProcessor

            proc = AutoProcessor.from_pretrained(repo)
            model = AutoModel.from_pretrained(repo).eval()
            inputs = proc(
                text=["a photo"], images=torch.zeros(3, 224, 224),
                return_tensors="pt", padding="max_length",
            )
            with torch.no_grad():
                out = model(**inputs)
            print(f"  {name:10s} OK   logits {tuple(out.logits_per_image.shape)}")
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:10s} FAIL {type(e).__name__}: {e}")
            missing.append(name)

    print(f"\n可用 {len(ok)}/8：{', '.join(ok)}")
    if missing:
        print(f"缺席 {len(missing)}：{', '.join(missing)}  ← 須在報告中明列")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
