"""命名規範：baseline 的結果樹與影像產物樹對應，參數掃描依 `<編輯器>/<變因>/` 分層。

1. 移動版控的結果表（`git mv`）：
   - `results/ultraedit/edits/<條件>.csv` → `results/ultraedit/edits_<條件>.csv`
   - `results/sweeps/<編輯器>/<名稱>.csv` → `results/sweeps/<編輯器>/<變因>/<名稱>.csv`（對照見 SWEEPS）
2. 改寫所有結果表中以 `artifacts/` 起首的路徑欄，前綴對照見 `artifact_prefixes()`；
   逐表驗證列數、欄序與非路徑欄逐值不變，路徑欄只有前綴改變。
3. 印出遠端要在 `baseline/` 執行的 `mv` 指令（`--remote-commands`）。

用法（repo 根）
    python archive/migration/restructure_result_layout.py --check            # 只列出將改動的檔案與格數
    python archive/migration/restructure_result_layout.py                    # 執行
    python archive/migration/restructure_result_layout.py --remote-commands  # 印出遠端指令
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "baseline"

#: (編輯器, 原名稱) → 變因。原名稱同時是 results 的檔名主幹與 artifacts 的目錄名。
SWEEPS = {
    ("flux", "guidance_2p0"): "guidance",
    ("flux", "guidance_3p5"): "guidance",
    ("flux", "true_cfg_3p5"): "true_cfg",
    ("ip2p", "reference_edits"): "reference",
    ("ip2p", "reference_off_target"): "reference",
    ("sdedit", "sd15_guidance"): "guidance",
    ("sdedit", "sd15_strength"): "strength",
    ("sdedit", "sd21_base_strength"): "strength",
    ("sdedit", "sd21_v_prediction"): "prediction_type",
    ("sdxl_ip2p", "guidance_portrait_pair"): "guidance",
    ("sdxl_ip2p", "guidance_portraits"): "guidance",
    ("sdxl_ip2p", "guidance_portraits_off_target"): "guidance",
    ("sdxl_ip2p", "high_image_guidance"): "image_guidance",
    ("sdxl_ip2p", "high_image_guidance_off_target"): "image_guidance",
    ("ultraedit", "image_guidance"): "image_guidance",
    ("ultraedit", "text_image_guidance"): "text_image_guidance",
    ("ultraedit", "noun_placement_variants"): "noun_placement",
    ("ultraedit", "noun_placement_variants_off_target"): "noun_placement",
    ("ultraedit", "prompt_templates_man_00_woman_00"): "prompt_template",
    ("ultraedit", "prompt_templates_man_00_woman_00_off_target"): "prompt_template",
    ("ultraedit", "prompt_templates_man_01_woman_01"): "prompt_template",
    ("ultraedit", "prompt_templates_man_01_woman_01_off_target"): "prompt_template",
}

DIRECTORIES = {
    "artifacts/flux_edits/": "artifacts/flux/edits/",
    "artifacts/ultraedit_edits/": "artifacts/ultraedit/edits/",
    "artifacts/defenses_aligned/": "artifacts/aligned/defenses/",
    "artifacts/defended_edits_aligned/": "artifacts/aligned/defended_edits/",
    "artifacts/purified_edits_aligned/": "artifacts/aligned/purified_edits/",
}


def artifact_prefixes() -> dict:
    out = dict(DIRECTORIES)
    for (editor, name), variable in SWEEPS.items():
        out[f"artifacts/sweeps/{editor}/{name}/"] = f"artifacts/sweeps/{editor}/{variable}/{name}/"
    return out


def moves() -> list:
    out = []
    for path in sorted((BASELINE / "results/ultraedit/edits").glob("*.csv")):
        out.append((path, BASELINE / "results/ultraedit" / f"edits_{path.stem}.csv"))
    for (editor, name), variable in SWEEPS.items():
        source = BASELINE / "results/sweeps" / editor / f"{name}.csv"
        if source.is_file():
            out.append((source, BASELINE / "results/sweeps" / editor / variable / f"{name}.csv"))
    return out


def rewrite(path: Path, prefixes: dict):
    data = path.read_bytes()
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    rows = list(reader)
    changed = 0
    new_rows = [rows[0]]
    for row in rows[1:]:
        new = []
        for cell in row:
            value = cell
            for old, replacement in prefixes.items():
                if value.startswith(old):
                    value = replacement + value[len(old):]
                    break
            changed += value != cell
            new.append(value)
        new_rows.append(new)
    if not changed:
        return None
    for old, new in zip(rows, new_rows):
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert a == b or (a.startswith("artifacts/") and a.split("/")[-1] == b.split("/")[-1])
    terminator = "\r\n" if data.split(b"\n", 1)[0].endswith(b"\r") else "\n"
    out = io.StringIO(newline="")
    csv.writer(out, lineterminator=terminator).writerows(new_rows)
    return out.getvalue().encode("utf-8"), changed


def remote_commands() -> None:
    print("cd ~/image-immunization/baseline")
    print("mkdir -p artifacts/flux artifacts/ultraedit artifacts/aligned")
    for old, new in DIRECTORIES.items():
        print(f"[ ! -e {old.rstrip('/')} ] || mv {old.rstrip('/')} {new.rstrip('/')}")
    print("# 等失真臂淨化後分母的相對連結多一層目錄，須重建")
    print("[ ! -L artifacts/aligned/purified_edits/undefended ] || "
          "ln -sfn ../../purified_edits/undefended artifacts/aligned/purified_edits/undefended")
    for (editor, name), variable in SWEEPS.items():
        old, new = f"artifacts/sweeps/{editor}/{name}", f"artifacts/sweeps/{editor}/{variable}/{name}"
        print(f"[ ! -e {old} ] || {{ mkdir -p artifacts/sweeps/{editor}/{variable} && mv {old} {new}; }}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--remote-commands", action="store_true")
    args = parser.parse_args()
    if args.remote_commands:
        remote_commands()
        return
    planned = moves()
    for source, target in planned:
        print(f"{source.relative_to(REPO)} → {target.relative_to(REPO)}")
    if not args.check:
        for source, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-C", str(REPO), "mv", str(source), str(target)], check=True)
    prefixes = artifact_prefixes()
    total = 0
    for path in sorted((REPO / "baseline/results").rglob("*.csv")):
        result = rewrite(path, prefixes)
        if result is None:
            continue
        data, changed = result
        total += changed
        print(f"{path.relative_to(REPO)}：{changed} 格")
        if not args.check:
            path.write_bytes(data)
    print(f"路徑格合計 {total}")


if __name__ == "__main__":
    main()
