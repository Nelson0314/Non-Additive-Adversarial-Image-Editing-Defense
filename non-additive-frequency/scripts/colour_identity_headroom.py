"""色彩載體對身分讀數的可達範圍：把擴散模型整條拿掉，直接問判準動不動得了。

為什麼要有這一支
────────────────────────────────────────────────────────────────────
`docs/DIRECTION.md` §6.0 把色彩載體判為「身分讀數下未淨化時就沒有效果」，
依據是三批經過 IP2P 編輯之後的量測。那條鏈上有五個環節（色彩映射 → VAE 編碼
→ UNet 去噪 → VAE 解碼 → 人臉嵌入），任何一環都可能是效果消失的地方，而那三批
分不出是哪一環。

本支把中間三環全部拿掉，只留頭尾：

    x  ──色彩映射──▶  x_col  ──人臉嵌入──▶  cos( E(x), E(x_col) )

於是它回答一個乾淨的問題：**在這個參數化與這個半徑下，色彩映射最多能把身分
餘弦壓到多低。** 這是色彩載體在身分軸上的**可達上界**。

**這一支用身分餘弦當損失，那不是本專案的目標函數。** `docs/GOAL.md` 的主讀數
是位移，語意指標並列；身分是後來為了分辨「劣化」與「重畫」才加的一個讀數，
從來不是規定的最佳化目標（損失軸那四個臂正好量到：把身分寫進損失反而更差，
+0.087 對代理損失的 +0.630）。

所以這一支能回答的只有兩件事，兩件都與「判準是什麼」無關：

1. **這個參數化學不學得動**（最佳化解對同半徑隨機解拉不拉得開）；
2. **`sign` 是不是瓶頸**（換 `adam` 差多少）。

**它不回答「色彩載體有沒有效」。** 那個問題要看四個讀數並列的表
（`runs/readout_parallel/README.md`），而那張表不需要重跑任何訓練。

**這個上界不是一個可以拿去用的工作點。** 要達到它就必須讓色彩映射蓋過臉本身，
而那是防禦方自己把主體毀掉——`runs/ip2p_decoy*` 那一族已經記過同型的事
（代理讀數看不出「防禦方自己把攻擊做完了」）。本支的用途是**診斷**：
若上界很高，那麼色彩族在既有三批裡量到的零就不是「判準對色彩不敏感」，
而是「凍結主體之後只剩經由 UNet 的間接路徑，而那條路徑很弱」。

兩篇被移植的論文（AdvCF, arXiv:2011.06690；AdvCF-Film, arXiv:2209.02430）
攻擊的都是**分類器**，判準是 top-1 標籤翻轉；本專案的判準是**人臉嵌入的餘弦**，
而人臉辨識器正是被訓練成對光照與色彩平衡不變的。兩者是不是同一件事，這一支
直接量。

三個變因，一次問完
────────────────────────────────────────────────────────────────────
`docs/DIRECTION.md` §3.4 列了三個還沒分開檢定的解釋，本支涵蓋其中兩個，
另加一個由程式碼比對發現的第三個：

| 變因 | 兩個值 | 為什麼要問 |
|---|---|---|
| `update` | `sign`／`adam` | AdvCF 走正規化梯度下降，本專案**全部**色彩批次走 `sign`（逐格核對過 `results.csv` 的 `update` 欄）。192 維的曲線空間裡各座標敏感度差幾個數量級，sign 丟掉梯度大小 |
| `param` | `color_grid`（本專案主線）／`color_curve`（AdvCF 原式） | 曲線族是逐行照抄 AdvCF `CF()` 的那一支 |

外加 `rand` 臂（角點隨機、零最佳化），因為本專案的色彩批次一再量到隨機解
在好幾欄上不輸最佳化解。

起點是這個損失的駐點——`--init` 存在的理由
────────────────────────────────────────────────────────────────────
損失是 `cos(E(x), E(x_def))`，而本專案每個參數化在零半徑時都**逐位元恆等**
（刻意的設計）。於是第 0 步 `x_def = x`，餘弦取到它的**全域極大 1.0**，
而極大點的梯度**恰為零**。`sign(0) = 0`，PGD 一步也走不動，逐步損失上只看到
「很平」。

這與 `patch_param._init_content` 記的是同一族陷阱（`edit_divergence` 在
`x_def = x` 處值與梯度都恰為零）。實測（`identity` 起點、400 步、11 張影像）：

| 參數化 | 最佳化臂裡「一步都沒動」的格數 |
|---|---|
| `color_grid`（49 152 參數） | **0 / 176** |
| `color_curve`（192 參數） | **160 / 176** |

`color_grid` 全部逃得出來，`color_curve` 只有一張影像逃得出來。差別是浮點：
`cos(a, a)` 在浮點下是 `1.0 ± 1e-7` 而不是恰好 1，殘留的雜訊夠不夠給出一組
可用的符號，與參數量有關。**也就是說 `identity` 起點下這一支是靠運氣的**，
`color_curve` 那 160 格不是讀數。

故 `--init random`（預設）由 `*RandomParam` 的角點抽樣給起點，繞開駐點；
`--init identity` 保留，用來重現那個陷阱本身。**兩種起點的結果不可並列。**

無論哪種起點，`stuck_at_init` 欄逐格記下「參數一步都沒動」，不靠人看 `linf`。

`outside` 那個臂**已經拿掉，理由要寫下來**
────────────────────────────────────────────────────────────────────
原本還有一個 `outside` 臂：色彩映射套在臉框之外、框內凍結，用來問「本專案的
`apply_where` 是不是效果消失的原因」。它跑出來每一格都是 `cos = 1.0000`、
第 81 步早停。

**那個讀數是設計的重言，不是結果。** 主讀數 `cos_fixed_box` 只看臉框裡的
像素，而 `outside` 臂被禁止改動的正是同一個框——輸出在框內逐位元等於原圖，
餘弦當然是 1。它量不到「衣物上的色彩經由 UNet 影響編輯輸出的臉」那條路，
而那才是真正的問題。要問那條路就必須把擴散模型放回去，也就是既有的
`ip2p_colour_*` 三批。

留著它只會產生一堆看起來像結論的 1.0000。

可微分的人臉裁切
────────────────────────────────────────────────────────────────────
`src/metrics/identity.embed` 走 PIL 與 `@torch.no_grad()`，梯度過不去。本支
另建一條可微分的路徑，並且**把它與 `embed` 的一致性當成一個 CSV 欄位量出來**
（`verify_cos`），不用推論的：

1. 人臉框由 `identity.face_boxes` 在**原圖**上取一次就固定。色彩映射是逐點的，
   不搬動任何像素，所以框在防禦圖上仍然對；把框固定住也讓「身分掉了」不會
   混進「偵測框跑掉了」。
2. 邊界與取整逐行取自 `facenet_pytorch.models.utils.detect_face.extract_face`
   的 margin 計算與 `int(max(...))` 夾取。
3. 縮放用 `F.interpolate(..., mode="bilinear", antialias=True)`。**這與原式
   不同**：`extract_face` 對 PIL 輸入走 `Image.resize(..., BILINEAR)`。兩者
   都是有抗鋸齒的雙線性，但濾波核的支撐計算不完全相同，故標 `modified`。
   實際偏差由 `verify_cos` 逐張量出。
4. 標準化 `(x·255 − 127.5)/128`，與 `MTCNN(post_process=True)` 的
   `fixed_image_standardization` 相同。

輸出
────────────────────────────────────────────────────────────────────
`runs/colour_identity_headroom/results.csv`，一格一列。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defense import color_param as cp  # noqa: E402
from src.metrics import identity as idm  # noqa: E402

FACE_SIZE = idm.DETECTOR_IMAGE_SIZE
FACE_MARGIN = idm.DETECTOR_MARGIN


def load_image(path: Path, device) -> torch.Tensor:
    """PNG → (1,3,H,W)，值域 [0,1]。"""
    from PIL import Image
    import numpy as np

    arr = np.asarray(Image.open(path).convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def margin_box(box: Tuple[float, float, float, float], h: int, w: int
               ) -> Tuple[int, int, int, int]:
    """`extract_face` 的 margin 與夾取，逐行照抄。回傳 (x0, y0, x1, y1)。"""
    mx = FACE_MARGIN * (box[2] - box[0]) / (FACE_SIZE - FACE_MARGIN)
    my = FACE_MARGIN * (box[3] - box[1]) / (FACE_SIZE - FACE_MARGIN)
    return (int(max(box[0] - mx / 2, 0)),
            int(max(box[1] - my / 2, 0)),
            int(min(box[2] + mx / 2, w)),
            int(min(box[3] + my / 2, h)))


def largest_box(x01: torch.Tensor, device
                ) -> Optional[Tuple[int, int, int, int]]:
    """最大的一個人臉框（加了 margin、已取整）；偵測不到回 `None`。

    取最大的那一個，是因為多人合照的身分讀數本來就不可靠
    （`HANDOFF.md` 第三坑），這一支只問「主要那張臉」。
    """
    boxes = idm.face_boxes(x01, device=device)
    if not boxes:
        return None
    h, w = x01.shape[-2:]
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return margin_box(boxes[-1], h, w)


def face_crop(x01: torch.Tensor, box: Tuple[int, int, int, int]
              ) -> torch.Tensor:
    """(1,3,H,W) → (1,3,160,160)，**可微分**，已做 fixed standardization。"""
    x0, y0, x1, y1 = box
    patch = x01[..., y0:y1, x0:x1]
    face = F.interpolate(patch, size=(FACE_SIZE, FACE_SIZE),
                         mode="bilinear", align_corners=False, antialias=True)
    return (face * 255.0 - 127.5) / 128.0


def build_param(kind: str, radius: float, apply_where: Optional[torch.Tensor]):
    """本專案主線的 `color_grid` G=16 D=8，與 AdvCF 原式的 `color_curve` K=64。"""
    if kind == "color_grid":
        return cp.ColorGridParam(radius=radius, grid=16, luma_bins=8,
                                 apply_where=apply_where)
    if kind == "color_curve":
        return cp.ColorCurveParam(radius=radius, pieces=64,
                                  bound_mode="symmetric",
                                  apply_where=apply_where)
    raise ValueError(f"未知的參數化 {kind!r}")


def outside_face_mask(x01: torch.Tensor, box: Tuple[int, int, int, int]
                      ) -> torch.Tensor:
    """(1,1,H,W)，臉框內為 0（凍結）、框外為 1。

    這是本專案 `apply_where` 那個約束的**最寬鬆**形式：真實的載體只有衣物
    （約 23%），這裡放行框外的全部像素。要問的是「凍結臉」這件事本身的代價，
    所以不能再疊上載體面積的限制，否則兩個變因混在一起。
    """
    w = torch.ones_like(x01[:, :1])
    x0, y0, x1, y1 = box
    w[..., y0:y1, x0:x1] = 0.0
    return w


# 各參數化的可學張量叫什麼。**明寫而不是從 `params()` 推**，理由見
# `run_cell` 裡 `post_reset` 的註解：隨機對照的 `params()` 是空的。
PARAM_TENSOR_ATTR = {"color_grid": "a", "color_curve": "theta"}


def build_rand(kind: str, radius: float, apply_where: Optional[torch.Tensor]):
    """角點隨機對照。`corner` 是本模組的預設，理由見 `color_param` 的註解。"""
    if kind == "color_grid":
        return cp.ColorGridRandomParam(radius=radius, grid=16, luma_bins=8,
                                       apply_where=apply_where, draw="corner")
    if kind == "color_curve":
        return cp.ColorCurveRandomParam(radius=radius, pieces=64,
                                        bound_mode="symmetric",
                                        apply_where=apply_where, draw="corner")
    raise ValueError(f"未知的參數化 {kind!r}")


def run_cell(x01, box, ref_embed, ref_embed_full, net, kind, radius, where,
             update, steps, seed, init) -> dict:
    """跑一格，回傳要寫進 CSV 的那一列。

    損失就是**判準本身**：`cos( E(原圖), E(色彩圖) )`，要最小化。這一支不用
    任何代理量——`runs/latent_norm_probe/` 已經量到色彩族裡 `latent_norm` 與
    效果是反相關的，這裡沒有理由再引入一個代理。
    """
    from src.defense.param_pgd import run_param_pgd

    aw = None if where == "whole" else outside_face_mask(x01, box)

    def loss_fn(x_def: torch.Tensor) -> torch.Tensor:
        e = net(face_crop(x_def, box))[0]
        return F.cosine_similarity(ref_embed[None], e[None])[0]

    if update == "rand":
        param = build_rand(kind, radius, aw)
        param.reset(x01, seed)
        with torch.no_grad():
            x_def = param.render(x01).clamp(0, 1)
        stopped_at, stop_reason = 0, "no_params"
    else:
        param = build_param(kind, radius, aw)
        post_reset = None
        if init == "random":
            # 起點由**同族的隨機參數化**給，不自己再寫一份抽樣：兩份抽樣一旦
            # 分岔，「最佳化解對隨機解」比較到的就混著起點的差異。
            # `run_param_pgd` 的 `post_reset` 恰好在 `reset()` 之後、第一步
            # 之前呼叫一次，是唯一寫得進參數張量的位置。
            seed_param = build_rand(kind, radius, aw)
            seed_param.reset(x01, seed)
            # **不可以走 `params()` 拿隨機起點。** `*RandomParam` 是不最佳化的
            # 對照條件，它的 `params()` **回傳空串列**（`NO_OPT_CONDS` 的設計）。
            # 走 `zip(pm.params(), seed_param.params())` 會安靜地複製零個張量，
            # 起點原封不動留在恆等上，而每一格看起來只是「這個半徑沒有效果」。
            # 已經踩過一次。故改用明寫的屬性名，並在下面用不變量檢查釘住。
            attr = PARAM_TENSOR_ATTR[kind]
            seed_val = getattr(seed_param, attr).detach().clone()

            def post_reset(pm, _v=seed_val, _a=attr):
                with torch.no_grad():
                    getattr(pm, _a).copy_(_v)

        res = run_param_pgd(x01, param, loss_fn, steps=steps, seed=seed,
                            update=update, eval_fn=loss_fn,
                            eval_every=max(1, steps // 40),
                            patience=8, min_delta=1e-4,
                            post_reset=post_reset)
        x_def = res.x_def.clamp(0, 1)
        stopped_at, stop_reason = res.stopped_at, res.stop_reason

    with torch.no_grad():
        # 主讀數：可微分路徑上的餘弦（框固定，與最佳化看到的是同一個量）。
        cos_fixed = float(F.cosine_similarity(
            ref_embed[None], net(face_crop(x_def, box))[0][None])[0])
        # 併列讀數：走 `identity.embed` 的正規路徑，**框重新偵測**。
        # 兩者分歧的原因只有兩個（框跑掉、或裁切核不同），而 `face_found`
        # 這一欄把前者分出來——那是本專案的主讀數認定的最強成功形式。
        e_def = idm.embed(x_def, device=x01.device)
        cos_detect = idm.similarity(ref_embed_full, e_def) if e_def is not None else None
        delta = (x_def - x01)
        linf = float(delta.abs().max())
        rmse = float((delta ** 2).mean().sqrt())
    # **駐點守門。** `linf == 0` 表示參數一步都沒動，那不是「這個半徑沒有
    # 效果」而是「梯度恰為零」。兩者在 `cos` 欄上長得一模一樣，故必須另立
    # 一欄，不可靠人去看 `linf`。
    stuck = update != "rand" and linf == 0.0
    if stuck and init == "random":
        # 隨機起點下**不可能**一步都沒動：角點抽樣在任何半徑上都給出非零的
        # 偏離，所以第 0 步的輸出就不等於原圖。走到這裡表示起點根本沒有寫進去
        # （例如又從 `params()` 拿），那是設定錯誤不是讀數。**寧可拋錯。**
        raise RuntimeError(
            f"{kind} r={radius} {update}：隨機起點下參數一步都沒動。"
            f"起點沒有寫進 {PARAM_TENSOR_ATTR[kind]}，檢查 post_reset。")
    return {
        "param": kind, "radius": radius, "where": where, "update": update,
        "init": init, "stuck_at_init": stuck,
        "steps": steps, "stopped_at": stopped_at, "stop_reason": stop_reason,
        "cos_fixed_box": round(cos_fixed, 5),
        "cos_redetect": None if cos_detect is None else round(cos_detect, 5),
        "face_found": e_def is not None,
        "linf": round(linf, 5), "rmse": round(rmse, 5),
    }


def verify_pipeline(x01, box, net) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """可微分路徑與 `identity.embed` 的一致性。

    回傳 `(可微分路徑的嵌入, embed 的嵌入, 兩者餘弦)`。**那個餘弦逐張寫進
    CSV**，不寫成註解——`CLAUDE.md`：每個未載的參數都要成為欄位。它偏離 1.0
    的部分全部來自縮放核的差異（本檔頭第 3 點），是這一支唯一的移植偏差。
    """
    with torch.no_grad():
        e_diff = net(face_crop(x01, box))[0]
        e_ref = idm.embed(x01, device=x01.device)
    if e_ref is None:
        raise RuntimeError("原圖偵測得到框卻取不到嵌入——兩者用同一個偵測器，"
                           "不該發生")
    return e_diff, e_ref, float(F.cosine_similarity(e_diff[None], e_ref[None])[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True,
                    help="影像目錄名（相對於 --data-root），一個目錄一張同名 PNG")
    ap.add_argument("--data-root", default="data/omniedit150")
    ap.add_argument("--out", default="runs/colour_identity_headroom")
    ap.add_argument("--params", nargs="+", default=["color_grid", "color_curve"])
    ap.add_argument("--radii", nargs="+", type=float,
                    default=[0.10, 0.30, 0.60, 1.20, 2.00])
    ap.add_argument("--arms", nargs="+",
                    default=["whole:sign", "whole:adam", "whole:rand"],
                    help="`where:update`。`where` 只有 `whole` 有意義，"
                         "`outside` 是設計上的重言（見檔頭），保留旗標只為了"
                         "讓那個判斷可以被重跑驗證。update 取 sign／adam／rand")
    ap.add_argument("--init", choices=("random", "identity"), default="random",
                    help="最佳化的起點。random（預設）由同族隨機參數化的角點"
                         "抽樣給，繞開「恆等起點是損失駐點」那個陷阱；"
                         "identity 保留原行為，用來重現那個陷阱本身")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    _, net = idm._load(device)

    rows: List[dict] = []
    skipped: List[str] = []
    for name in args.images:
        path = ROOT / args.data_root / name / f"{name}.png"
        if not path.exists():
            raise FileNotFoundError(f"找不到影像 {path}")
        x01 = load_image(path, device)
        box = largest_box(x01, device)
        if box is None:
            # 偵測不到臉的影像**不是一個讀數**，是這一支問不了的影像：
            # 沒有 `E(原圖)` 就沒有餘弦可比。記下來，不要靜默跳過。
            skipped.append(name)
            print(f"[skip] {name}：原圖偵測不到人臉", flush=True)
            continue
        e_diff, e_ref, vcos = verify_pipeline(x01, box, net)
        print(f"[{name}] box={box} verify_cos={vcos:.5f}", flush=True)
        for kind in args.params:
            for arm in args.arms:
                where, update = arm.split(":")
                for radius in args.radii:
                    row = run_cell(x01, box, e_diff, e_ref, net, kind, radius,
                                   where, update, args.steps, args.seed,
                                   args.init)
                    row.update(image=name, verify_cos=round(vcos, 5),
                               box="_".join(str(v) for v in box))
                    rows.append(row)
                    print(f"  {kind:11s} {arm:12s} r={radius:<5.2f} "
                          f"cos={row['cos_fixed_box']:.4f} "
                          f"redetect={row['cos_redetect']} "
                          f"stop={row['stop_reason']}@{row['stopped_at']}"
                          f"{'  <- 駐點，一步沒動' if row['stuck_at_init'] else ''}",
                          flush=True)

    if not rows:
        raise RuntimeError("沒有任何一張影像偵測得到人臉，這一批問不了問題")
    cols = ["image", "param", "radius", "where", "update", "init",
            "stuck_at_init", "steps",
            "stopped_at", "stop_reason", "cos_fixed_box", "cos_redetect",
            "face_found", "linf", "rmse", "verify_cos", "box"]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n寫出 {len(rows)} 列到 {out_dir / 'results.csv'}")
    n_stuck = sum(1 for r in rows if r["stuck_at_init"])
    n_opt = sum(1 for r in rows if r["update"] != "rand")
    # **收工時印出來，不要只留在 CSV 裡。** 這個失效不會讓任何一格報錯，
    # 而 `cos = 1.0000` 讀起來就像「這個半徑沒有效果」。
    print(f"最佳化臂 {n_opt} 格，其中 {n_stuck} 格參數一步都沒動"
          f"（stuck_at_init）。那些格不是讀數。")
    if skipped:
        print(f"偵測不到人臉而略過：{' '.join(skipped)}")


if __name__ == "__main__":
    main()
