"""實驗進度儀表板（唯讀）— `docs/reference/CODE_CONTRACTS.md` §2.4。

給實驗 agent 監察格點狀態用。**唯讀、不碰 GPU、不啟停任何行程、
不寫入批次目錄**（`--html` 產生的 `dashboard.html` 除外）。

    python scripts/dashboard.py <批次目錄> [選項]

      --watch [秒]   輪詢並重印，預設 30 秒
      --html         產生 <批次>/dashboard.html
      --json         單行機器可讀摘要（agent 一律用這個）
      --failed       只列失敗的格，附 error 全文
      --rebuild      忽略 progress.json，掃描 _cells/ 重建

## 給 agent 的約定

- 輪詢一律用 `--json`，不要解析終端表格。
- 看到 `failed > 0` 時用 `--failed` 取完整錯誤，**回報使用者，不要自行重跑**。
  先驗實驗十次事後診斷都是「一個值被沿用到另一個對象上而且沒有症狀」，
  盲目重跑會把症狀蓋掉。
- `skipped` 不是失敗，是結構上不適用（例如 APA 移植在低 τ 上低於 VAE 重建下限）。
- 不要在 GPU 工作執行期間跑 pytest 或其他 CPU 密集工作：實測會把單張 SDEdit
  由 222 s 拉長到 30 分鐘以上。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.progress import STATES, read_progress, rebuild  # noqa: E402

STAGE_ORDER = ("calib", "train", "rayscale", "eval", "report")


def _hms(seconds) -> str:
    if not seconds:
        return "-"
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _stage_key(name: str):
    """依流程順序排，未知的段落排在最後但保持穩定順序。"""
    return (STAGE_ORDER.index(name) if name in STAGE_ORDER else len(STAGE_ORDER),
            name)


def load(batch_dir: Path, use_rebuild: bool) -> dict:
    if use_rebuild:
        return rebuild(batch_dir)
    try:
        return read_progress(batch_dir)
    except FileNotFoundError:
        # 快取不存在時退回真相來源。這不是掩蓋症狀——progress.json 本來就
        # 只是快取，_cells/ 才是真相（見 src/utils/progress.py 模組 docstring）。
        return rebuild(batch_dir)


def render_text(snap: dict) -> str:
    env = snap.get("env") or {}
    head = (f"batch {snap.get('batch', '?')}   "
            f"gpu={env.get('gpu', '?')}  precision={env.get('precision', '?')}  "
            f"commit={env.get('commit', '?')}")
    lines = [head, ""]
    lines.append(f"{'stage':<11}{'total':>6}{'done':>7}{'run':>5}{'fail':>6}"
                 f"{'skip':>6}{'pending':>9}{'elapsed':>10}{'eta':>10}")
    lines.append("-" * 70)
    for name in sorted(snap.get("stages", {}), key=_stage_key):
        st = snap["stages"][name]
        elapsed = sum(c.get("seconds") or 0 for c in snap.get("cells", [])
                      if c.get("stage") == name)
        lines.append(
            f"{name:<11}{st['total']:>6}{st['done']:>7}{st['running']:>5}"
            f"{st['failed']:>6}{st['skipped']:>6}{st['pending']:>9}"
            f"{_hms(elapsed):>10}{_hms(st.get('eta_seconds')):>10}"
        )
    lines.append("-" * 70)
    s = snap.get("summary", {})
    lines.append(
        f"{'TOTAL':<11}{s.get('total', 0):>6}{s.get('done', 0):>7}"
        f"{s.get('running', 0):>5}{s.get('failed', 0):>6}{s.get('skipped', 0):>6}"
        f"{s.get('pending', 0):>9}{_hms(s.get('elapsed_seconds')):>10}"
        f"{_hms(s.get('eta_seconds')):>10}"
    )

    failed = [c for c in snap.get("cells", []) if c.get("status") == "failed"]
    if failed:
        lines.append("")
        for c in failed[:10]:
            first = (c.get("error") or "").strip().splitlines()
            lines.append(f"FAILED  {c['id']}   {first[0] if first else ''}")
        if len(failed) > 10:
            lines.append(f"        …另有 {len(failed) - 10} 格失敗，用 --failed 看全部")
        lines.append("")
        lines.append("失敗的格請回報使用者，不要自行重跑。")
    return "\n".join(lines)


def render_failed(snap: dict) -> str:
    failed = [c for c in snap.get("cells", []) if c.get("status") == "failed"]
    if not failed:
        return "沒有失敗的格。"
    out = []
    for c in failed:
        out.append(f"=== {c['id']}")
        out.append(f"    config_hash: {c.get('config_hash', '?')}")
        out.append(f"    started    : {c.get('started', '?')}")
        out.append("    error      :")
        out.extend("      " + ln for ln in (c.get("error") or "").splitlines())
        out.append("")
    return "\n".join(out)


def render_json(snap: dict) -> str:
    """單行機器可讀。欄位固定，agent 依此判斷是否該介入。"""
    stages = {n: {k: st.get(k, 0) for k in STATES} | {"total": st["total"]}
              for n, st in snap.get("stages", {}).items()}
    return json.dumps({
        "batch": snap.get("batch"),
        "updated": snap.get("updated"),
        "env": snap.get("env"),
        "summary": snap.get("summary"),
        "stages": stages,
    }, ensure_ascii=False)


def render_html(snap: dict) -> str:
    rows = []
    for name in sorted(snap.get("stages", {}), key=_stage_key):
        st = snap["stages"][name]
        pct = 100.0 * st["done"] / st["total"] if st["total"] else 0.0
        rows.append(
            f"<tr><td>{name}</td><td>{st['total']}</td><td>{st['done']}</td>"
            f"<td>{st['running']}</td>"
            f"<td class='{'bad' if st['failed'] else ''}'>{st['failed']}</td>"
            f"<td>{st['skipped']}</td><td>{st['pending']}</td>"
            f"<td><div class='bar'><i style='width:{pct:.1f}%'></i></div></td></tr>"
        )
    failed = [c for c in snap.get("cells", []) if c.get("status") == "failed"]
    fail_html = "".join(
        f"<details><summary>{c['id']}</summary><pre>{c.get('error', '')}</pre></details>"
        for c in failed
    )
    env = snap.get("env") or {}
    return f"""<title>progress {snap.get('batch', '')}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:60rem}}
table{{border-collapse:collapse;width:100%}}
th,td{{padding:.35rem .6rem;border-bottom:1px solid #8884;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.bad{{color:#c33;font-weight:600}}
.bar{{background:#8882;height:.7rem;border-radius:.35rem;overflow:hidden;min-width:6rem}}
.bar i{{display:block;height:100%;background:#4a9}}
pre{{white-space:pre-wrap;background:#8881;padding:.6rem;border-radius:.3rem}}
@media(prefers-color-scheme:dark){{body{{background:#111;color:#ddd}}}}
</style>
<h1>batch {snap.get('batch', '')}</h1>
<p>gpu={env.get('gpu', '?')} · precision={env.get('precision', '?')} ·
   commit={env.get('commit', '?')} · updated {snap.get('updated', '')}</p>
<table><tr><th>stage</th><th>total</th><th>done</th><th>run</th><th>fail</th>
<th>skip</th><th>pending</th><th>progress</th></tr>{''.join(rows)}</table>
{'<h2>failed</h2>' + fail_html if failed else ''}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="實驗進度儀表板（唯讀）")
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--watch", nargs="?", type=int, const=30, default=None,
                    metavar="秒")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--failed", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)

    if not args.batch_dir.exists():
        print(f"批次目錄不存在：{args.batch_dir}", file=sys.stderr)
        return 2

    def once() -> int:
        snap = load(args.batch_dir, args.rebuild)
        if args.json:
            print(render_json(snap))
        elif args.failed:
            print(render_failed(snap))
        else:
            print(render_text(snap))
        if args.html:
            out = args.batch_dir / "dashboard.html"
            out.write_text(render_html(snap), encoding="utf-8")
            if not args.json:
                print(f"\n已寫出 {out}")
        # 離開碼讓 shell 條件判斷可用：有失敗即非零
        return 1 if snap.get("summary", {}).get("failed", 0) else 0

    if args.watch is None:
        return once()
    try:
        while True:
            print("\033[2J\033[H", end="")     # 清畫面，讓輪詢輸出不會愈疊愈長
            once()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
