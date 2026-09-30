"""`scripts/evaluate_condition.sh` 各階段的完成標記：綁定解析後設定、輸入雜湊與輸出驗收。

階段（`<條件>` 為 `configs/conditions.yaml` 的條件名）

    defense                防禦圖：`generate_color_defenses` 以條件參數解析後的全部設定；輸入為資料集影像、
                           遮罩、`prompts.yaml` 與 LPIPS 參照表
    edit_ip2p              防禦後編輯：`pipelines.editing` 解析後的設定（seed、步數、s_t、s_i、影像子集等）；
                           輸入為防禦圖
    purify                 七道淨化：淨化協定 `purifiers/protocol.json`；輸入為防禦圖
    pedit_<淨化>_ip2p      淨化後編輯：同 edit_ip2p；輸入為該道淨化的輸出

標記檔 `runtime/state/<條件>.<階段>.done` 的內容是上述內容的 SHA-256 摘要。`check` 只在標記內容等於
重新計算的摘要、且該階段的輸出通過驗收時回傳 0；設定、輸入或輸出任一不符即回傳 1，由呼叫端重跑。
`write` 先驗收輸出，通過才寫入標記。

用法（color 專案根）
    python -m immunization_color.stages check <條件> <階段> [--images <影像>...]
    python -m immunization_color.stages write <條件> <階段> [--images <影像>...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from immunization_color import conditions, layout
from immunization_color.cli import evaluate_queue_job as validation

DATA = "data/portraits"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root: Path, paths) -> dict:
    out = {}
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError(f"缺少輸入：{path}")
        out[path.relative_to(root).as_posix()] = sha256(path)
    if not out:
        raise ValueError("階段沒有任何輸入檔")
    return out


def plain(namespace: argparse.Namespace, drop) -> dict:
    return {k: (v.as_posix() if isinstance(v, Path) else v)
            for k, v in sorted(vars(namespace).items()) if k not in drop}


def condition_args(arm: str) -> list:
    if arm not in conditions.CONDITIONS:
        raise ValueError(f"未知的條件：{arm}")
    return conditions.CONDITIONS[arm]


def purifier_labels() -> list:
    from immunization_core.purifiers.protocol import purifier_labels as labels
    return labels(include_identity=False)


def edit_argv(arm: str, tag: str, images) -> list:
    if tag == "edit_ip2p":
        source, out, suffix = f"artifacts/defenses/{arm}", f"artifacts/defended_edits/{arm}", f"_{arm}"
    else:
        purifier = tag[len("pedit_"):-len("_ip2p")]
        source = f"artifacts/purified/{arm}/{purifier}"
        out, suffix = f"artifacts/purified_edits/{arm}/{purifier}", f"_{arm}_{purifier}"
    argv = ["--data-root", DATA, "--defenses-dir", source, "--output-dir", out,
            "--scenarios", "ip2p", "--suffix", suffix]
    return argv + (["--images", *images] if images else [])


def stage_record(project: Path, arm: str, tag: str, images=None) -> dict:
    """該階段的設定與輸入雜湊；摘要即此紀錄的 JSON 之 SHA-256。"""
    data = project / DATA
    defenses = project / "artifacts/defenses" / arm
    if tag == "defense":
        from immunization_color.cli.generate_color_defenses import build_parser
        args = build_parser().parse_args(["--output-dir", "unused", "--arm", arm,
                                          *condition_args(arm)])
        inputs = [*data.glob("*/*.png"), data / "prompts.yaml"]
        lpips_ref = args.lpips_ref
        if lpips_ref.is_absolute() and lpips_ref.is_relative_to(layout.PROJECT):
            lpips_ref = project / lpips_ref.relative_to(layout.PROJECT)  # 預設值為專案根下的絕對路徑
        elif not lpips_ref.is_absolute():
            lpips_ref = project / lpips_ref
        return {"stage": tag, "settings": plain(args, {"out", "data", "images", "lpips_ref"}),
                "inputs": hashes(project, [*inputs, lpips_ref])}
    defended = sorted(defenses.glob(f"*__{arm}__def.png"))
    if tag == "purify":
        import immunization_core
        protocol = Path(immunization_core.__file__).parent / "purifiers/protocol.json"
        return {"stage": tag, "settings": json.loads(protocol.read_text(encoding="utf-8")),
                "inputs": hashes(project, defended)}
    if tag == "edit_ip2p" or (tag.startswith("pedit_") and tag.endswith("_ip2p")):
        from immunization_core.pipelines.editing import build_parser
        argv = edit_argv(arm, tag, images)
        args = build_parser().parse_args(argv)
        source = project / argv[argv.index("--defenses-dir") + 1]
        inputs = defended if tag == "edit_ip2p" else sorted(source.glob("*.png"))
        return {"stage": tag, "settings": plain(args, {"data", "out", "defended"}),
                "inputs": hashes(project, [*inputs, data / "prompts.yaml"])}
    raise ValueError(f"未知階段：{tag}")


def digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_outputs(project: Path, arm: str, tag: str) -> None:
    defenses = project / "artifacts/defenses" / arm
    if tag == "defense":
        names = validation.defense_names(defenses, arm)
        validation.validate_table(defenses / "results.csv", ("image",), {(n,) for n in names})
    elif tag == "edit_ip2p":
        validation.edit_keys(project, arm, project / "artifacts/defended_edits" / arm, f"_{arm}")
    elif tag == "purify":
        names = validation.defense_names(defenses, arm)
        for purifier in purifier_labels():
            for name in names:
                validation.artifact(project / "artifacts/purified" / arm / purifier / f"{name}__def.png")
    else:
        purifier = tag[len("pedit_"):-len("_ip2p")]
        validation.edit_keys(project, arm, project / "artifacts/purified_edits" / arm / purifier,
                             f"_{arm}_{purifier}")


def marker(project: Path, arm: str, tag: str) -> Path:
    return project / "runtime/state" / f"{arm}.{tag}.done"


def check(project: Path, arm: str, tag: str, images=None) -> tuple:
    """(是否可略過, 原因)。"""
    path = marker(project, arm, tag)
    if not path.is_file():
        return False, "沒有完成標記"
    try:
        current = digest(stage_record(project, arm, tag, images))
        validate_outputs(project, arm, tag)
    except (ValueError, FileNotFoundError) as problem:  # 輸出或輸入不齊：需重跑
        return False, str(problem)
    if path.read_text(encoding="utf-8").strip() != current:
        return False, "設定或輸入與完成標記不符"
    return True, ""


def write(project: Path, arm: str, tag: str, images=None) -> None:
    validate_outputs(project, arm, tag)
    path = marker(project, arm, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest(stage_record(project, arm, tag, images)) + "\n", encoding="utf-8")


def chain_tags() -> list:
    return ["defense", "edit_ip2p", "purify", *[f"pedit_{p}_ip2p" for p in purifier_labels()]]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "write"))
    parser.add_argument("arm")
    parser.add_argument("stage")
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--project-root", dest="project", type=Path, default=layout.PROJECT)
    args = parser.parse_args(argv)
    if args.action == "check":
        ok, reason = check(args.project, args.arm, args.stage, args.images)
        if not ok:
            print(f"[RERUN] {args.arm}/{args.stage}：{reason}", file=sys.stderr)
        raise SystemExit(0 if ok else 1)
    try:
        write(args.project, args.arm, args.stage, args.images)
    except (ValueError, FileNotFoundError) as problem:
        raise SystemExit(f"[FATAL] {args.arm}/{args.stage} 的輸出未通過驗收：{problem}")


if __name__ == "__main__":
    main()
