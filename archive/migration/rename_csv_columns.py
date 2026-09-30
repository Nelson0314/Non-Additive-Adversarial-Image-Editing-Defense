"""第 9 項：CSV 欄名與識別值的英式拼法改為美式，不保留兩種 schema。

欄名對照（表頭中完全相符的欄才改；完整清單見 COLUMNS）
    defence_png           → defense_png
    defence               → defense
    deltaE00*             → delta_e00*（含原 deltaE00_skin_colour）
    D_／P_／DT_lpips_*    → disp_／predicted_disp_／residual_disp_lpips_*
    D_csv、siglip_pair_T、blocked_T → displacement_csv_lpips_full、siglip_pair_residual、blocked_residual

值對照（只改指定欄內完全相符的字串片段）
    solver_prompt_source：`diffvax.py::immunise` → `diffvax.py::immunize`
    spec_source：`../scripts/` → `archive/anti-purification/scripts/`

只改寫表頭與上述欄位；其餘位元組不變。每份表驗證列數、欄數、未改欄位逐值相同，
任一不符即中止且不寫任何檔。

用法（repo 根）
    python archive/migration/rename_csv_columns.py baseline color style
    python archive/migration/rename_csv_columns.py --check baseline color style
遠端 artifacts 內的 CSV（例如 `preflight.csv`）以同一支腳本處理，參數給該目錄。
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

COLUMNS = {"defence_png": "defense_png", "defence": "defense",
           "deltaE00_skin_colour": "delta_e00_skin_color",
           # 命名規範：欄名一律小寫 snake_case，不用 D、P、D_T 這類代號。
           "deltaE00": "delta_e00", "deltaE00_frame": "delta_e00_frame",
           "deltaE00_face_box": "delta_e00_face_box", "deltaE00_skin_color": "delta_e00_skin_color",
           "D_csv": "displacement_csv_lpips_full",
           "siglip_pair_T": "siglip_pair_residual", "blocked_T": "blocked_residual",
           **{f"{old}_lpips_{region}": f"{new}_lpips_{region}"
              for old, new in (("D", "disp"), ("P", "predicted_disp"), ("DT", "residual_disp"))
              for region in ("full", "subject", "background")}}
VALUES = {"solver_prompt_source": [("diffvax.py::immunise", "diffvax.py::immunize")],
          "spec_source": [("../scripts/", "archive/anti-purification/scripts/")]}


def parse(data: bytes):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    return rows[0], rows[1:]


def terminator(data: bytes) -> str:
    end = data.find(b"\n")
    return "\r\n" if end > 0 and data[end - 1:end] == b"\r" else "\n"


def rewrite(path: Path):
    """回傳 (新內容, 改動欄名, 改動值數)；不需改寫時回傳 None。"""
    data = path.read_bytes()
    header, rows = parse(data)
    new_header = [COLUMNS.get(c, c) for c in header]
    targets = {i: VALUES[c] for i, c in enumerate(header) if c in VALUES}
    changed_values = 0
    new_rows = []
    for row in rows:
        row = list(row)
        for i, pairs in targets.items():
            if i < len(row):
                value = row[i]
                for old, new in pairs:
                    value = value.replace(old, new)
                changed_values += value != row[i]
                row[i] = value
        new_rows.append(row)
    renamed = [(a, b) for a, b in zip(header, new_header) if a != b]
    if not renamed and not changed_values:
        return None
    if len(set(new_header)) != len(new_header):
        raise SystemExit(f"{path}：改名後欄名重複 {new_header}")
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator=terminator(data))
    if targets and changed_values:
        writer.writerow(new_header)
        writer.writerows(new_rows)
        result = out.getvalue().encode("utf-8")
    else:
        writer.writerow(new_header)
        first = data.find(b"\n") + 1
        result = out.getvalue().encode("utf-8") + data[first:]
    check_header, check_rows = parse(result)
    assert check_header == new_header and len(check_rows) == len(rows), path
    for old, new in zip(rows, check_rows):
        assert len(old) == len(new), path
        for i, (a, b) in enumerate(zip(old, new)):
            assert a == b or i in targets, (path, i)
    return result, renamed, changed_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="只檢查，仍有舊欄名或舊值即結束碼 1")
    args = parser.parse_args()
    plans = {}
    for root in args.roots:
        for path in sorted(root.rglob("*.csv")):
            if "vendor" in path.parts:
                continue
            result = rewrite(path)
            if result:
                plans[path] = result
    for path, (_, renamed, values) in plans.items():
        print(f"{path.as_posix()}：{renamed or ''} {f'值 {values} 格' if values else ''}".rstrip())
    if args.check:
        raise SystemExit(1 if plans else 0)
    for path, (data, _, _) in plans.items():
        path.write_bytes(data)
    print(f"[DONE] 改寫 {len(plans)} 份表")


if __name__ == "__main__":
    main()
