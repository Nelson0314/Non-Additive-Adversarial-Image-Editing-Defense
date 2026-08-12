"""把任意來源影像正規化成 `data/lo_aligned/<類別>/*.png`，並留下出處紀錄。

    python scripts/prepare_dataset.py --src ~/Downloads/raw --dst data/lo_aligned
    python scripts/prepare_dataset.py --dst data/lo_aligned --check

來源影像請按類別放進子目錄，類別名稱必須與 `prompts.yaml` 的鍵一致：

    raw/
      man/    *.jpg
      woman/  *.jpg
      dog/    *.png
      ...

處理內容（全部是決定性的，同一張輸入永遠得到同一張輸出）：

1. 轉 RGB。帶 alpha 的來源直接丟掉 alpha 而非合成到白底——合成會在物件
   邊緣造出人工的高對比邊，那正好是注意力與銳利度指標最敏感的地方。
2. **置中裁切成正方形後**再縮放。直接縮放會改變長寬比，而
   `aggregate_token_attention` 只支援方形格點（見該函式的說明）；更重要的是
   非等比縮放會系統性改變影像的空間頻率內容，鈍化與注意力兩軸都會受影響。
3. 以 LANCZOS 縮放到 `--size`（預設 512，與 SD v1.4 的訓練解析度一致）。
4. 存成 PNG。**不可用 JPEG**：Lo 的論文自己在限制那節指出 JPEG 壓縮可以
   消掉免疫擾動，資料集本身就帶壓縮痕跡會讓那條軸的量測失去基準。

出處紀錄寫在 `<dst>/provenance.json`：每張輸出對應的來源檔名、原始尺寸、
來源檔的 SHA-256。`runs/` 是唯一的證據來源而實驗無法重跑（見 CLAUDE.md），
資料集的可追溯性必須跟結果一起入版控。

`--check` 不處理任何影像，只驗證現有資料集：類別是否齊全、每類張數、尺寸
是否一致、prompts.yaml 是否每類都有 content。跑實驗前應該先過這一關。
"""

import argparse
import hashlib
import json
from pathlib import Path

VALID_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec(dst: Path) -> dict:
    import yaml

    pf = dst / "prompts.yaml"
    if not pf.exists():
        raise SystemExit(
            f"找不到 {pf}。資料集的類別、編輯 prompt 與要保護的內容 c_a "
            "都由該檔定義，沒有它無法決定要處理哪些類別"
        )
    spec = yaml.safe_load(pf.read_text(encoding="utf-8")) or {}
    for cls, entry in spec.items():
        if not isinstance(entry, dict) or "content" not in entry:
            raise SystemExit(
                f"{pf} 的類別 {cls!r} 缺少 content 欄。semantic attack 需要"
                "「要保護哪個詞」，那不能由 prompt 推測"
            )
        if not entry.get("prompts"):
            raise SystemExit(f"{pf} 的類別 {cls!r} 沒有任何編輯 prompt")
    return spec


def prepare(src: Path, dst: Path, size: int, spec: dict) -> list:
    from PIL import Image

    records = []
    for cls in spec:
        sdir = src / cls
        if not sdir.is_dir():
            print(f"  [略過] {sdir} 不存在")
            continue
        ddir = dst / cls
        ddir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p for p in sdir.iterdir() if p.suffix.lower() in VALID_SUFFIX
        )
        for i, p in enumerate(files):
            img = Image.open(p)
            w, h = img.size
            img = img.convert("RGB")
            # 置中裁切成正方形，再縮放。理由見模組 docstring 第 2 點。
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
            img = img.resize((size, size), Image.LANCZOS)
            out = ddir / f"{cls}_{i:02d}.png"
            img.save(out, format="PNG")
            records.append({
                "output": str(out.relative_to(dst)).replace("\\", "/"),
                "source_name": p.name,
                "source_sha256": sha256(p),
                "source_size": [w, h],
                "crop_side": side,
                "output_size": size,
            })
            print(f"  {p.name}  {w}×{h} → {size}² → {out.name}")
    return records


def check(dst: Path, spec: dict) -> int:
    from PIL import Image

    problems = 0
    print(f"{'類別':<10}{'張數':<6}{'content':<12}{'尺寸'}")
    print("-" * 48)
    for cls, entry in spec.items():
        d = dst / cls
        pngs = sorted(d.glob("*.png")) if d.is_dir() else []
        sizes = {Image.open(p).size for p in pngs}
        size_txt = ", ".join(f"{w}×{h}" for w, h in sorted(sizes)) or "—"
        print(f"{cls:<10}{len(pngs):<6}{entry['content']:<12}{size_txt}")
        if not pngs:
            print(f"  [缺] {cls} 沒有任何影像")
            problems += 1
        if len(sizes) > 1:
            print(f"  [不一致] {cls} 有多種尺寸，指標不可比")
            problems += 1
    prov = dst / "provenance.json"
    if not prov.exists():
        print(f"\n[缺] {prov} 不存在；資料集無法追溯來源")
        problems += 1
    else:
        n = len(json.loads(prov.read_text(encoding="utf-8")))
        total = sum(len(list((dst / c).glob('*.png')))
                    for c in spec if (dst / c).is_dir())
        print(f"\nprovenance.json 記錄 {n} 筆，實際 {total} 張")
        if n != total:
            print("  [不符] 出處紀錄與實際張數不一致")
            problems += 1
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="來源根目錄，底下按類別分子目錄")
    ap.add_argument("--dst", default="data/lo_aligned")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--check", action="store_true",
                    help="只驗證現有資料集，不處理任何影像")
    args = ap.parse_args()

    dst = Path(args.dst)
    spec = load_spec(dst)

    if args.check:
        n = check(dst, spec)
        raise SystemExit(0 if n == 0 else f"\n共 {n} 個問題")

    if not args.src:
        raise SystemExit("需要 --src；只想驗證請用 --check")
    records = prepare(Path(args.src), dst, args.size, spec)
    if not records:
        raise SystemExit(
            "沒有處理到任何影像。請確認 --src 底下的子目錄名稱與 "
            "prompts.yaml 的類別一致"
        )
    (dst / "provenance.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(records)} 張，出處寫入 {dst/'provenance.json'}")


if __name__ == "__main__":
    main()
