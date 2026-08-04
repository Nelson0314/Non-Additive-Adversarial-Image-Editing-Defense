"""把報告 HTML 裡的 `src="ASSET:<相對路徑>[|<最大寬度>][|jpeg]"` 換成內嵌影像。

    python scripts/build_report_html.py docs/REPORT_2026-08-04.src.html \
                                        docs/REPORT_2026-08-04.html

產出的檔案不依賴 repo 的目錄結構，可單獨寄出。影像一律由 `runs/` 取，
不使用示意圖——`runs/` 是唯一的證據來源。

找不到來源影像時直接拋出。以佔位圖或空白代替會讓報告看起來完整而實際上
少了證據，那與本專案反覆記錄的「無症狀缺陷」同型。
"""

import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(r'src="ASSET:([^"]+)"')


def encode(spec: str) -> str:
    parts = spec.split("|")
    rel = parts[0]
    max_w = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    fmt = parts[2].lower() if len(parts) > 2 else "png"

    src = ROOT / rel
    if not src.exists():
        raise SystemExit(f"找不到影像：{src}")
    im = Image.open(src).convert("RGB")
    if max_w and im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)

    buf = io.BytesIO()
    if fmt == "jpeg":
        im.save(buf, "JPEG", quality=92, subsampling=0, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, "PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    print(f"  {rel}  {im.width}x{im.height}  {len(b64) / 1024:.0f} KB")
    return f'src="data:{mime};base64,{b64}"'


def main():
    if len(sys.argv) != 3:
        raise SystemExit("用法：build_report_html.py <來源.html> <輸出.html>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    html = src.read_text(encoding="utf-8")
    print(f"[report] 內嵌影像：")
    out = PAT.sub(lambda m: encode(m.group(1)), html)
    dst.write_text(out, encoding="utf-8")
    print(f"[report] 寫出 {dst}（{len(out.encode('utf-8')) / 1024 / 1024:.2f} MB）")


if __name__ == "__main__":
    main()
