"""派工腳本的換行必須是 LF。

存在理由
────────────────────────────────────────────────────────────────────
遠端跑的是 bash，`\r` 會被當成指令的一部分，於是

    set -uo pipefail        →  set: pipefail: invalid option name
    case "$1" in            →  syntax error near unexpected token $'in\r'

而**本機完全看不出來**：Git Bash 讀得動、`bash -n` 在某些版本上也過得去、
編輯器不顯示。實際發生過：`scripts/patch_res_round.sh` 帶著 CRLF 上傳，
整批在遠端燒測時死掉、佇列中止，而前一批已經跑了三個半小時。

**來源是 Windows 上的 Python。** `pathlib.Path.write_text()` 預設
`newline=None`，寫入時把 `\n` 換成 `os.linesep`，在 Windows 上就是 `\r\n`。
用它改任何一個 `.sh` 都會靜默地把整個檔案變成 CRLF。
改 shell 腳本一律用 heredoc，或明給 `newline="\n"`。

`.py` 不受影響（Python 自己讀得動 CRLF），故本測試只管 `.sh`。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_所有派工腳本都是LF():
    bad = []
    for p in sorted((ROOT / "scripts").glob("*.sh")):
        if b"\r\n" in p.read_bytes():
            bad.append(p.name)
    assert not bad, (
        "這些腳本含 CRLF，上傳後在遠端會死於語法錯誤："
        + " ".join(bad)
        + "。用 `perl -pi -e 's/\r\n/\n/g' <檔>` 修，"
          "並改用 heredoc 而不是 Python 的 write_text 產生 .sh。")


def test_腳本第一行是有效的shebang():
    """CRLF 最先弄壞的就是 shebang（`#!/usr/bin/env bash\r`），
    症狀是 `bad interpreter`，與換行完全聯想不起來。"""
    bad = []
    for p in sorted((ROOT / "scripts").glob("*.sh")):
        first = p.read_bytes().split(b"\n", 1)[0]
        if not first.startswith(b"#!") or first.endswith(b"\r"):
            bad.append(p.name)
    assert not bad, "shebang 缺失或帶 CR：" + " ".join(bad)


def test_sync_check_有連線守門():
    """`sync_check.sh` 在 ssh 失敗時必須拒絕判定，不可回報「全部同步」。

    原本的寫法：ssh 連不上時遠端 md5 清單只剩一行錯誤訊息，於是
    `MISSING` 空、`join` 比不出東西、`BAD` 空，腳本印出「✓ 全部同步」。
    **伺服器連不上被回報成完全同步**——比漏傳更糟，因為它看起來是通過的。
    實測發生過：遠端斷線十一小時，這支仍然說全部同步。

    這一條釘住兩件事：所有 ssh 取值都走同一個函式，而那個函式檢查回傳碼。
    """
    src = (ROOT / "scripts" / "sync_check.sh").read_text(encoding="utf-8")
    assert "fetch_remote_sums()" in src, "取遠端 md5 的邏輯必須集中成一個函式"
    body = src.split("fetch_remote_sums()", 1)[1]
    assert "if ! ssh" in body, "必須檢查 ssh 的回傳碼"
    assert "不判定同步狀態" in body, "失敗時要明說不判定，不可默默當成同步"
    # 除了函式裡那一次，不應該再有別的 ssh 取值
    assert src.count("ssh -o BatchMode") == 1, (
        "還有別處直接呼叫 ssh 取 md5，那一處會繞過守門")
