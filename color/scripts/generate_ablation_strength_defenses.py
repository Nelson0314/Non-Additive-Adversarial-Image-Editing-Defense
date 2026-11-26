"""消融降強度：B2（色相旋轉＋飽和度縮放）與 B3（3×3 仿射）在較低防禦圖 LPIPS 目標下的求解。

條件名 color_baseline_<huesat|affine>_lpips<r>_pexels，r 以 p 表示小數（0p85 ＝ 0.85）。
目標 t ＝ r · t_Full，t_Full 為 Full（color3d_scale2p5_pexels）同一張的防禦圖 LPIPS（trial results/color3d_scale2p5_pexels/fidelity.csv，
8 位元量化後的防禦圖以 piq.LPIPS 量得）。其餘全部沿用 trial generate_color_baseline_defenses.py：同一載體與參數尺度、
同一求解參數（conds/color3d_orig.args 經 solver_args 刪去 DROP 清單；PHR、Adam 1800 步、lr 3e-4 → 1e-5、lut-init 0.002）、
同一區間約束（with_lpips_interval：上限 t ＋ 0.015、下界 t − 0.015、soft 上限 t；只交付量化後落在區間內的檢查點，
1800 步內沒有即不存檔，記為 failed_no_feasible_checkpoint）。本檔只換 t。
由 trial run_lut_queue.sh 經 conds/<條件>.generator 呼叫（--arm、--images、--data-root、--output-root）；
需 PYTHONPATH 含 trial 根目錄（run_lut_queue.sh 的 setup_env 已設定）。
"""
import re
import sys

import generate_color_baseline_defenses as matched  # trial 模組：登錄 B2／B3 載體與 color3d 目標

NAME = re.compile(r"^(color_baseline_(?:huesat|affine))_lpips(\d+)p(\d+)_pexels$")


def parse(arm):
    m = NAME.match(arm)
    if not m:
        raise SystemExit(f"條件名不符 color_baseline_<huesat|affine>_lpips<r>_pexels：{arm}")
    return m.group(1), float(f"{m.group(2)}.{m.group(3)}")


if __name__ == "__main__":
    arm = sys.argv[sys.argv.index("--arm") + 1]
    carrier, ratio = parse(arm)
    color3d, base = matched.color3d, matched.base
    color3d.TARGETS[arm.removesuffix("_pexels")] = color3d.TARGETS["color3d_scale2p5"]
    color3d.SmoothAngleLut = matched.CARRIERS[carrier]
    goal = {img: ratio * t for img, t in matched.targets().items()}
    run = base.main

    def main_with_target(argv):
        img = argv[argv.index("--images") + 1]
        print(f"[STRENGTH] {arm} {img} ratio {ratio} target {goal[img]:.4f}", flush=True)
        return run(matched.with_lpips_interval(argv, goal[img]))

    base.main = main_with_target
    color3d.main()
