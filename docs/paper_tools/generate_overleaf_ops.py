"""Word-level edit operations that turn the Overleaf text of one file into the repo text.

    python docs/paper_tools/generate_overleaf_ops.py <overleaf_file> <repo_file> <ops.json>

<overleaf_file> is the current Overleaf text (from the project zip or a copy of the editor text);
<repo_file> is the target. Output: {"file", "len", "ops": [[offset, deleted, inserted], ...]} with offsets in
UTF-16 code units of the Overleaf text, plus the FNV-1a hashes of both texts, which the editor-side snippet in
README.md checks before and after applying the operations. Applying the operations inside the Overleaf editor keeps
the edit history; uploading or replacing files does not.
"""
import difflib
import json
import re
import sys
from pathlib import Path

TOKEN = re.compile(r"\s+|\w+|[^\w\s]")


def u16(s):
    return len(s.encode("utf-16-le")) // 2


def fnv(s):
    data = s.encode("utf-16-le")
    h = 0x811C9DC5
    for i in range(0, len(data), 2):
        h = ((h ^ int.from_bytes(data[i:i + 2], "little")) * 0x01000193) & 0xFFFFFFFF
    return h


def main():
    src, dst, out = (Path(p) for p in sys.argv[1:4])
    a = src.read_text(encoding="utf-8").replace("\r\n", "\n")
    b = dst.read_text(encoding="utf-8").replace("\r\n", "\n")
    ta, tb = TOKEN.findall(a), TOKEN.findall(b)
    offs = [0]
    for t in ta:
        offs.append(offs[-1] + u16(t))
    ops = [[offs[i1], "".join(ta[i1:i2]), "".join(tb[j1:j2])]
           for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes() if tag != "equal"]
    out.write_text(json.dumps({"file": dst.name, "len": u16(a), "ops": ops}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(ops)} ops; before hash {fnv(a)} len {u16(a)}; after hash {fnv(b)} len {u16(b)}")


if __name__ == "__main__":
    main()
