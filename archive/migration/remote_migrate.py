"""Move the remote artifacts from the pre-restructure tree into a fresh clone.

Usage (on the lab server, from any directory):
    python3 remote_migrate.py --old ~/image-immunization --new ~/image-immunization.new [--apply]

Without --apply only the plan is printed. Every move is a rename on the same
NFS mount. A destination that already exists aborts the move (whole-directory
moves) or is compared byte by byte (archive merge: identical files stay in the
old tree, differing files are reported and also stay). Symlinks in the old
tree are not followed; the two lab links to baseline edits are replaced by
copies so that each project owns its inputs.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
from pathlib import Path

BASELINE = [
    ("runs/defence_portraits", "baseline/artifacts/defenses"),
    ("runs/eps_aligned", "baseline/artifacts/defenses_aligned"),
    ("runs/edit_defended_aligned", "baseline/artifacts/defended_edits_aligned"),
    ("runs/edit_purified_aligned", "baseline/artifacts/purified_edits_aligned"),
    ("runs/edit_preflight", "baseline/artifacts/undefended_edits"),
    ("runs/edit_defended", "baseline/artifacts/defended_edits"),
    ("runs/edit_purified", "baseline/artifacts/purified_edits"),
    ("runs/purified", "baseline/artifacts/purified"),
    ("main_table/images/flux_full", "baseline/artifacts/flux_edits"),
    ("main_table/images/ultraedit_full/edit_preflight", "baseline/artifacts/ultraedit_edits/undefended_edits"),
    ("main_table/images/ultraedit_full/edit_defended", "baseline/artifacts/ultraedit_edits/defended_edits"),
    ("main_table/images/ultraedit_full/edit_purified", "baseline/artifacts/ultraedit_edits/purified_edits"),
    ("main_table/images/flux_preview", "baseline/artifacts/sweeps/flux/guidance_3p5"),
    ("main_table/images/flux_preview_g2", "baseline/artifacts/sweeps/flux/guidance_2p0"),
    ("main_table/images/flux_preview_truecfg", "baseline/artifacts/sweeps/flux/true_cfg_3p5"),
    ("main_table/images/sdedit_preview", "baseline/artifacts/sweeps/sdedit/sd15_strength"),
    ("main_table/images/sdedit_preview_sd21", "baseline/artifacts/sweeps/sdedit/sd21_v_prediction"),
    ("main_table/images/sdedit_preview_sd21base_sweep", "baseline/artifacts/sweeps/sdedit/sd21_base_strength"),
    ("main_table/images/sdedit_guidance_sweep_sd15", "baseline/artifacts/sweeps/sdedit/sd15_guidance"),
    ("main_table/images/sd_family/sd3_ultraedit_quick", "baseline/artifacts/sweeps/ultraedit/image_guidance"),
    ("main_table/images/sd_family/sd3_ultraedit_low_guidance", "baseline/artifacts/sweeps/ultraedit/text_image_guidance"),
    ("main_table/images/sd_family/ultraedit_prompt_round2", "baseline/artifacts/sweeps/ultraedit/noun_placement_variants"),
    ("main_table/images/sd_family/ultraedit_prompt_sweep_a", "baseline/artifacts/sweeps/ultraedit/prompt_templates_man_00_woman_00"),
    ("main_table/images/sd_family/ultraedit_prompt_sweep_b", "baseline/artifacts/sweeps/ultraedit/prompt_templates_man_01_woman_01"),
    ("main_table/images/sd_family/sdxl_ip2p_grid", "baseline/artifacts/sweeps/sdxl_ip2p/guidance_portrait_pair"),
    ("main_table/images/sd_family/sdxl_ip2p_all8", "baseline/artifacts/sweeps/sdxl_ip2p/guidance_portraits"),
    ("main_table/images/sd_family/sdxl_ip2p_high_image_guidance", "baseline/artifacts/sweeps/sdxl_ip2p/high_image_guidance"),
]
COLOR = [
    ("lab/runs/defence", "color/artifacts/defenses"),
    ("lab/runs/defence_shards", "color/artifacts/defense_shards"),
    ("lab/runs/edit_defended", "color/artifacts/defended_edits"),
    ("lab/runs/purified", "color/artifacts/purified"),
    *[(f"lab/runs/edit_purified/{c}", f"color/artifacts/purified_edits/{c}")
      for c in ("color", "color_simple", "color_simple_skinbox", "color_simple_xattn")],
    ("lab/runs/logs", "color/runtime/logs"),
]
STYLE = [(f"lab/runs/style_prompt_{r}{suffix}", f"style/artifacts/{kind}/{r}")
         for r in ("r11", "r13", "cls_p_noedit", "cls_p_snow")
         for suffix, kind in (("", "defenses"), ("_edit", "edits"))]
# Inputs another project reads: copied, never linked across projects.
COPIES = [
    ("baseline/artifacts/undefended_edits", "color/artifacts/undefended_edits"),
    ("baseline/artifacts/purified_edits/undefended", "color/artifacts/purified_edits/undefended"),
    ("baseline/artifacts/undefended_edits/ip2p_si18", "style/artifacts/undefended_edits/ip2p_si18"),
]
# Remaining pre-restructure research tree, merged file by file into the archive.
ARCHIVE = [(name, f"archive/anti-purification/{name}")
           for name in ("runs", "data", "configs", "docs", "scripts", "src", "tests")]


def move_dir(old: Path, new: Path, src: str, dst: str, apply: bool, log: list) -> None:
    s, d = old / src, new / dst
    if s.is_symlink() or not s.is_dir():
        log.append(f"[MISSING] {src}")
        return
    if d.exists():
        raise SystemExit(f"[ABORT] 目的地已存在：{dst}")
    log.append(f"[MOVE] {src} -> {dst}")
    if apply:
        d.parent.mkdir(parents=True, exist_ok=True)
        os.rename(s, d)


def copy_dir(new: Path, src: str, dst: str, apply: bool, log: list) -> None:
    s, d = new / src, new / dst
    if d.exists():
        raise SystemExit(f"[ABORT] 目的地已存在：{dst}")
    log.append(f"[COPY] {src} -> {dst}")
    if apply:
        if not s.is_dir():
            raise SystemExit(f"[ABORT] 複製來源不存在：{src}")
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(s, d, symlinks=False)


def merge_tree(old: Path, new: Path, src: str, dst: str, apply: bool, log: list, stats: dict) -> None:
    s_root, d_root = old / src, new / dst
    if not s_root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(s_root, topdown=True):
        dirnames[:] = [n for n in dirnames if n not in ("__pycache__", ".pytest_cache")]
        rel = Path(dirpath).relative_to(s_root)
        target_dir = d_root / rel
        # A directory absent from the destination moves in one rename.
        for name in list(dirnames):
            sd, td = Path(dirpath) / name, target_dir / name
            if not td.exists() and not sd.is_symlink():
                stats["dirs"] += 1
                if apply:
                    td.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(sd, td)
                dirnames.remove(name)
        for name in filenames:
            sf, tf = Path(dirpath) / name, target_dir / name
            if sf.is_symlink():
                stats["links"] += 1
            elif not tf.exists():
                stats["files"] += 1
                if apply:
                    tf.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(sf, tf)
            elif filecmp.cmp(sf, tf, shallow=False):
                stats["identical"] += 1
            else:
                stats["differ"] += 1
                log.append(f"[DIFFER] {src}/{rel / name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    old, new = args.old.expanduser().resolve(), args.new.expanduser().resolve()
    if not (new / ".git").is_dir() or not (new / "archive/anti-purification/runs").is_dir():
        raise SystemExit(f"[ABORT] {new} 不是完整簽出的 git clone")
    log: list[str] = []
    for src, dst in BASELINE + COLOR + STYLE:
        move_dir(old, new, src, dst, args.apply, log)
    for src, dst in COPIES:
        copy_dir(new, src, dst, args.apply, log)
    stats = {"dirs": 0, "files": 0, "identical": 0, "differ": 0, "links": 0}
    for src, dst in ARCHIVE:
        merge_tree(old, new, src, dst, args.apply, log, stats)
    print("\n".join(log))
    print(f"[ARCHIVE] 整目錄搬入 {stats['dirs']}、檔案搬入 {stats['files']}、與版控相同留在舊樹 {stats['identical']}、"
          f"內容不同留在舊樹 {stats['differ']}、符號連結 {stats['links']}")
    print("[APPLIED]" if args.apply else "[DRY-RUN] 未移動任何檔案")


if __name__ == "__main__":
    main()
