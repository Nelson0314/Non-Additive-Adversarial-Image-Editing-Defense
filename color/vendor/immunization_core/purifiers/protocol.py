"""固定淨化協定的唯一正本讀取：identity 對照與七道淨化，順序與強度取自 `protocol.json`。

Python 以 `PURIFIERS`、`label()`、`purifier_labels()` 取用；shell 以
`python -m immunization_core.purifiers.protocol --exclude-identity` 取得空白分隔的標籤。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL_FILE = Path(__file__).with_name("protocol.json")


def label(kind: str, strength: float) -> str:
    return kind if not strength else f"{kind}{strength:g}"


def _load() -> tuple:
    spec = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    entries = tuple((item["kind"], item["strength"]) for item in spec["purifiers"])
    if not entries or entries[0][0] != "identity":
        raise ValueError(f"{PROTOCOL_FILE}：第一項必須是 identity 對照")
    labels = [label(*entry) for entry in entries]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{PROTOCOL_FILE}：標籤重複 {labels}")
    return entries


#: 逐項 (kind, strength)，第一項為 identity 對照。
PURIFIERS = _load()


def purifier_labels(include_identity: bool = True) -> list:
    return [label(kind, strength) for kind, strength in PURIFIERS
            if include_identity or kind != "identity"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exclude-identity", action="store_true")
    args = parser.parse_args(argv)
    print(" ".join(purifier_labels(not args.exclude_identity)))


if __name__ == "__main__":
    main()
