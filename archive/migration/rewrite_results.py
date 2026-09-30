"""第 7 項：改寫 baseline、color、style 結果 CSV 的路徑欄與識別值，並逐表驗證。

路徑欄依產物角色改寫為相對所屬專案根的新版面；舊值有四種形式（遠端 NFS 絕對、
Windows 絕對、主線根相對、lab 相對），先去除根再依前綴規則對應，未知前綴即失敗。
識別值只改寫 `colour_curve_ours` → `color_curve`；UltraEdit 的位移與保留率表在
`scenario` 後加入 `editor` 欄。其餘欄位逐位元保留。

用法（repo 根）
    python archive/migration/rewrite_results.py            # 改寫並寫出驗證報告
    python archive/migration/rewrite_results.py --check    # 只驗證，不寫檔（僅適用於改寫前）
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path

PATH_COLUMNS = ("data", "data_root", "input_png", "output_png", "png", "defended_png",
                "undefended_png", "source_png", "original_png", "defence_png", "reference_png")
ROOTS = ("/nfs/home/nelson0314/image-immunization/", "C:/image-immunization/anti-purification/",
         "C:/image-immunization/")

SWEEP_BATCHES = {
    "sd3_ultraedit_quick": "ultraedit/image_guidance",
    "sd3_ultraedit_low_guidance": "ultraedit/text_image_guidance",
    "ultraedit_prompt_round2": "ultraedit/noun_placement_variants",
    "ultraedit_prompt_sweep_a": "ultraedit/prompt_templates_man_00_woman_00",
    "ultraedit_prompt_sweep_b": "ultraedit/prompt_templates_man_01_woman_01",
    "sdxl_ip2p_grid": "sdxl_ip2p/guidance_portrait_pair",
    "sdxl_ip2p_all8": "sdxl_ip2p/guidance_portraits",
    "sdxl_ip2p_high_image_guidance": "sdxl_ip2p/high_image_guidance",
}

BASELINE_RULES = [
    ("runs/defence_portraits/", "artifacts/defenses/"),
    ("main_table/images/defence_portraits/", "artifacts/defenses/"),
    ("runs/eps_aligned/", "artifacts/defenses_aligned/"),
    ("runs/edit_defended_aligned/", "artifacts/defended_edits_aligned/"),
    ("runs/edit_purified_aligned/", "artifacts/purified_edits_aligned/"),
    ("runs/edit_preflight/", "artifacts/undefended_edits/"),
    ("runs/edit_defended/", "artifacts/defended_edits/"),
    ("runs/edit_purified/", "artifacts/purified_edits/"),
    ("runs/purified/", "artifacts/purified/"),
    ("runs/color_import/", "artifacts/color_import/"),
    ("main_table/images/flux_full/", "artifacts/flux_edits/"),
    ("main_table/images/ultraedit_full/edit_preflight/", "artifacts/ultraedit_edits/undefended_edits/"),
    ("main_table/images/ultraedit_full/edit_defended/", "artifacts/ultraedit_edits/defended_edits/"),
    ("main_table/images/ultraedit_full/edit_purified/", "artifacts/ultraedit_edits/purified_edits/"),
    ("main_table/images/flux_preview/", "artifacts/sweeps/flux/guidance_3p5/"),
    ("main_table/images/flux_preview_g2/", "artifacts/sweeps/flux/guidance_2p0/"),
    ("main_table/images/flux_preview_truecfg/", "artifacts/sweeps/flux/true_cfg_3p5/"),
    ("main_table/images/sdedit_preview/", "artifacts/sweeps/sdedit/sd15_strength/"),
    ("main_table/images/sdedit_preview_sd21/", "artifacts/sweeps/sdedit/sd21_v_prediction/"),
    ("main_table/images/sdedit_preview_sd21base_sweep/", "artifacts/sweeps/sdedit/sd21_base_strength/"),
    ("main_table/images/sdedit_guidance_sweep_sd15/", "artifacts/sweeps/sdedit/sd15_guidance/"),
    *[(f"main_table/images/sd_family/{old}/", f"artifacts/sweeps/{new}/")
      for old, new in SWEEP_BATCHES.items()],
    ("data/portraits/", "data/portraits/"),
]

LAB_RULES = [
    ("lab/data/portraits/", "data/portraits/"),
    ("lab/runs/defence/", "artifacts/defenses/"),
    ("lab/runs/edit_defended/", "artifacts/defended_edits/"),
    ("lab/runs/edit_purified/", "artifacts/purified_edits/"),
    ("lab/runs/purified/", "artifacts/purified/"),
    ("lab/runs/edit_preflight/", "artifacts/undefended_edits/"),
]
STYLE_RUN = re.compile(r"^lab/runs/style_prompt_([a-z0-9_]+?)(_edit)?/")

IDENTIFIERS = {"colour_curve_ours": "color_curve"}
EDITOR_TABLES = {"baseline/results/ultraedit/displacement.csv": "ultraedit",
                 "baseline/results/ultraedit/retention.csv": "ultraedit"}


def strip_root(value: str) -> str:
    for root in ROOTS:
        if value.startswith(root):
            return value[len(root):]
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"未知的絕對路徑根：{value}")
    return value


def map_path(value: str, project: str) -> str:
    if value == "":
        return value
    rel = strip_root(value.replace("\\", "/"))
    # 前綴比對以目錄邊界為準：暫加結尾斜線，輸出時移除。
    probe = rel + "/"
    out = None
    if project != "baseline" and (m := STYLE_RUN.match(probe)):
        base = "artifacts/edits/" if m.group(2) else "artifacts/defenses/"
        out = base + m.group(1) + "/" + probe[m.end():]
    else:
        for old, new in (BASELINE_RULES if project == "baseline" else LAB_RULES):
            if probe.startswith(old):
                out = new + probe[len(old):]
                break
    if out is None:
        raise ValueError(f"{project}：未知的路徑前綴 {value}")
    return out[:-1]


def map_identifiers(value: str) -> str:
    for old, new in IDENTIFIERS.items():
        value = re.sub(rf"\b{old}\b", new, value)
    return value


def read(path: Path):
    data = path.read_bytes()
    if b"\r\n" in data:
        raise ValueError(f"{path}: 預期 LF 行尾")
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    return data, rows


def render(rows) -> bytes:
    out = io.StringIO(newline="")
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue().encode("utf-8")


def rewrite(path: Path, relative: str):
    data, rows = read(path)
    if render(rows) != data:
        raise ValueError(f"{path}: csv 往返不是逐位元相同，不能安全改寫")
    project = relative.split("/")[0]
    header, body = rows[0], rows[1:]
    new_header = list(header)
    editor = EDITOR_TABLES.get(relative)
    if editor:
        new_header.insert(header.index("scenario") + 1, "editor")
    changed = {"path": 0, "identifier": 0}
    new_body = []
    for row in body:
        if len(row) != len(header):
            raise ValueError(f"{path}: 欄數不符")
        new = []
        for column, value in zip(header, row):
            if column in PATH_COLUMNS:
                mapped = map_path(value, project)
                changed["path"] += mapped != value
            else:
                mapped = map_identifiers(value)
                changed["identifier"] += mapped != value
            new.append(mapped)
            if column == "scenario" and editor:
                new.append(editor)
        new_body.append(new)
    return data, rows, [new_header] + new_body, changed


def verify(relative, old_rows, new_rows, manifest_entry):
    """列數、鍵集合、非改寫欄逐值相同；與遷移前 manifest 的表頭與列數一致。"""
    old_header, new_header = old_rows[0], new_rows[0]
    kept = [c for c in old_header if c not in PATH_COLUMNS]
    old_idx = {c: i for i, c in enumerate(old_header)}
    new_idx = {c: i for i, c in enumerate(new_header)}
    assert len(old_rows) == len(new_rows), relative
    assert [c for c in new_header if c != "editor" or relative not in EDITOR_TABLES] == old_header, relative
    key_columns = [c for c in ("condition", "arm", "scenario", "image", "prompt_index", "purifier",
                               "variant", "guidance_scale", "image_guidance_scale", "strength",
                               "style", "strength", "pairing") if c in old_idx]
    old_keys = sorted(tuple(map_identifiers(r[old_idx[c]]) for c in key_columns) for r in old_rows[1:])
    new_keys = sorted(tuple(r[new_idx[c]] for c in key_columns) for r in new_rows[1:])
    assert old_keys == new_keys, relative
    for old, new in zip(old_rows[1:], new_rows[1:]):
        for c in kept:
            if old[old_idx[c]] != new[new_idx[c]]:
                assert map_identifiers(old[old_idx[c]]) == new[new_idx[c]], (relative, c)
    result = {"rows": len(new_rows) - 1, "key_columns": key_columns}
    if manifest_entry is not None:
        assert manifest_entry["rows"] == len(old_rows) - 1, relative
        assert manifest_entry["header"] == old_header, relative
        result["manifest_path"] = manifest_entry["path"]
    return result


def origins(repo: Path) -> dict:
    """現行路徑 → 遷移前路徑（由 git 改名紀錄追溯）。"""
    out = subprocess.run(["git", "log", "--name-status", "-M", "--format=", "8bcaae0..HEAD", "--", "."],
                         cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    renames = {}
    for line in reversed(out.splitlines()):
        parts = line.split("\t")
        if parts[0].startswith("R") and len(parts) == 3:
            source, target = parts[1], parts[2]
            renames[target] = renames.pop(source, source)
    return renames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads((repo / "archive/migration/pre_migration_manifest.json").read_text(encoding="utf-8"))
    origin = origins(repo)
    report = {}
    for path in sorted(p for project in ("baseline", "color", "style")
                       for p in (repo / project / "results").rglob("*.csv")):
        relative = path.relative_to(repo).as_posix()
        data, old_rows, new_rows, changed = rewrite(path, relative)
        source = origin.get(relative, relative)
        entry = manifest["files"].get(source)
        if entry is not None:
            entry = dict(entry, path=source)
        report[relative] = {**verify(relative, old_rows, new_rows, entry), "changed": changed}
        if entry is not None:
            blob = subprocess.run(["git", "hash-object", "--stdin"], input=data, capture_output=True,
                                  check=True).stdout.decode().strip()
            if blob != entry["blob"]:
                raise ValueError(f"{relative}: 改寫前內容與遷移前 manifest 的 blob 不同")
            report[relative]["manifest_blob"] = blob
        if not args.check:
            path.write_bytes(render(new_rows))
    if not args.check:
        (repo / "archive/migration/csv_rewrite_report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
    totals = {k: sum(r["changed"][k] for r in report.values()) for k in ("path", "identifier")}
    print(f"{len(report)} 份 CSV；改寫路徑 {totals['path']} 格、識別值 {totals['identifier']} 格；"
          f"{sum('manifest_path' in r for r in report.values())} 份對上遷移前 manifest")


if __name__ == "__main__":
    main()
