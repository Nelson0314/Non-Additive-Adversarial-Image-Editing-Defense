"""量 A 段（DEC-016）壓下來的 φ=0 重建下限，並產逐圖比對頁。

要回答的問題只有一個：**把階段一搬到 VAE 之後，`φ=0` 的下限降到多少。**
故本腳本不跑訓練、不跑編輯、不跑 eval，只做四件事：

1. 量現行下限——先用 `decode(encode(x))`，再走一次真正的生成路徑
   （模塊停用的 BDIA 反演＋去噪）確認兩者一致。不一致就代表下限另有來源，
   那件事必須先看到再談壓縮
2. **A1**：解 `z*` 使 `decode(z*) ≈ x`
3. **A2**：固定 `z*`，只開解碼器的 GroupNorm affine 與各層 conv bias 對這張圖
   過擬合，達到目標即停（硬停止條件，見 `src/defense/recon.py`）
4. 逐圖存圖與指標，產一份**自足**的 `compare.html`（影像 base64 內嵌）

用法（旗標與批次 profile 共用一份，不手抄）
──────────────────────────────────────────────────────────────────────
    COMMON=$(BATCH=s3t20 bash scripts/shard.sh common)
    python scripts/recon_floor_ab.py --out ~/wacv_runs/fa_s3t20 \\
        --images horse_00 horse_03 woman_03 $COMMON

    COMMON=$(BATCH=ip20 bash scripts/shard.sh common)
    python scripts/recon_floor_ab.py --out ~/wacv_runs/fa_ip20 \\
        --images horse_00 man_00 bird_03 $COMMON

兩批跑完後合成一頁：

    python scripts/recon_floor_ab.py --page-only \\
        img2img=~/wacv_runs/fa_s3t20 inpainting=~/wacv_runs/fa_ip20 \\
        --page ~/wacv_runs/recon_floor_ab.html

產物（DEC-017：只留產出的影像與數值，逐步的中間圖不寫）
──────────────────────────────────────────────────────────────────────
逐影像五張 PNG（原圖／舊下限／A1／A1+A2／差分放大兩張）、`floor_ab.csv`、
`floor_ab.json`（逐步 history 與停止點）、`recon_<影像>.pt`（`z*` 與被微調的
解碼器參數，供 B／C 段接手）。逐步的中間影像一張都不寫。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch                                                 # noqa: E402

from src.defense import recon                                # noqa: E402
from src.defense.generator import DefenseGenerator           # noqa: E402
from src.experiment import executors                         # noqa: E402
from tau_preview import diff_map                             # noqa: E402

# 報表要看的四項。銳利度比不對稱（`MetricSuite.pairwise` 的 a 必須是原圖），
# 呼叫端一律以 `pairwise(x01, y)` 的順序取。
KEYS = ("lpips", "dists", "psnr", "acutance_ratio")

# 逐像素項的權重，**逐目標校準**。兩個感知量的量級差約三倍（本批六張影像的
# 重建下限實測 LPIPS 0.085–0.158、DISTS 0.023–0.050），沿用同一個權重會讓
# 換目標的同時也悄悄換掉了「感知項對逐像素項」的比例，而那個變因不會有症狀
# ——先驗紀錄裡重複十次的缺陷型態就是這個。0.15 是 0.5 依上述比例縮下來的。
# 未列出的目標直接拋出，不內插也不退回預設值。
W_PIXEL = {"lpips": 0.5, "dists": 0.15}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="產物目錄（量測模式必填）")
    ap.add_argument("--images", nargs="+", default=[])
    ap.add_argument("--a1-steps", type=int, default=300)
    ap.add_argument("--a1-lr", type=float, default=0.02)
    ap.add_argument("--a2-steps", type=int, default=300)
    ap.add_argument("--a2-lr", type=float, default=2e-3)
    ap.add_argument("--objective", choices=sorted(W_PIXEL), default="lpips",
                    help="壓哪一個下限。損失的感知項與停止判準一起換——"
                         "只換其中一個會得到「優化 A 卻以 B 判停」的組合，"
                         "而那在紀錄上看不出來")
    ap.add_argument("--w-pixel", type=float, default=None,
                    help="重建損失裡逐像素項的權重，感知項固定為 1。"
                         f"未給時依 --objective 查表 {W_PIXEL}")
    ap.add_argument("--gamma-acut", type=float, default=1.0,
                    help="銳利度 hinge 的係數。0 為關閉——關掉時 A1 會拿變鈍"
                         "換下限（實測銳利度比 0.9935 → 0.7887）。"
                         "本腳本無論如何都會另跑一欄 γ=0 作為對照")
    ap.add_argument("--acut-band", type=float, default=0.05,
                    help="銳利度帶的半寬 |1−銳利度比| 的容差。實際使用的帶是"
                         "它與該影像舊下限自身偏差取大者（`recon.acutance_band`）")
    ap.add_argument("--floor-ratio", type=float, default=0.50,
                    help="A2 的硬停止目標，取該影像舊下限（依 --objective）"
                         "的這個比例。預設 0.50 即先驗紀錄的 A1+A2 落點"
                         "（LPIPS 0.1434 → 0.0716）。它是停止規則，"
                         "不是對結果的宣稱")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resp-scale", type=float, default=0.05,
                    help="latent 反應探針的擾動能量，相對於 ‖z*‖")
    ap.add_argument("--resp-seed", type=int, default=20260810)
    ap.add_argument("--merge", nargs="+", default=[],
                    help="把逐影像分片的產物疊成一個目錄（第一個是目的地）。"
                         "檔名全部帶影像 id，故直接疊不會互相覆蓋")
    ap.add_argument("--page-only", nargs="+", default=[],
                    help="不載入模型，只把既有的產物目錄合成一頁。"
                         "格式 `標籤=路徑`")
    ap.add_argument("--page", type=Path, default=None,
                    help="`--page-only` 的輸出檔")
    return ap


def measure(args, rest) -> Path:
    # 批次設定不在此重建一份平行的參數表——那是兩份會分岔的清單——而是把未知
    # 旗標原樣轉交 `run_stage` 的 parser（`scripts/tau_preview.py` 同一作法）。
    import run_stage as rs

    argv = ["calib", "--batch", args.out.name, "--runs-root", str(args.out.parent),
            "--images", *args.images, *rest]
    rs_args = rs.build_parser().parse_args(argv)
    rs.resolve_thresholds(rs_args, verbose=False)
    res = rs.build_resources(rs_args, args.out)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    key = args.objective
    perceptual = {"lpips": res.suite.lpips_module,
                  "dists": res.suite.dists_module}[key]
    w_pixel = W_PIXEL[key] if args.w_pixel is None else args.w_pixel
    print(f"[目標] {key}（感知項與停止判準）  w_pixel={w_pixel}", flush=True)
    # 解碼器要開的那一組參數在整批影像上是同一組物件，取一次即可；數值的
    # 還原由 `recon.restored` 逐影像負責。
    tunable = recon.decoder_tunable(res.sd.vae.decoder)
    params = [p for _, p in tunable]
    n_tunable = sum(p.numel() for p in params)
    print(f"[A2] 可微調參數 {len(params)} 組共 {n_tunable} 個"
          f"（解碼器全參數 {sum(p.numel() for p in res.sd.vae.decoder.parameters())}）",
          flush=True)

    rows, detail = [], {}
    for image_id in args.images:
        entry = res.image(image_id)
        x = entry.x01

        def pair(y):
            m = res.suite.pairwise(x, y)
            return {k: m[k] for k in KEYS}

        # ---- 1. 現行下限 ----
        with torch.no_grad():
            y_vae = res.sd.decode_latent(res.sd.encode_image(x))
        m_vae = pair(y_vae)

        module = executors.build_module("apa", res, entry, seed=res.cfg.seed)
        try:
            module.disable()
            gen = DefenseGenerator(res.sd, module, k_inv=res.cfg.k_inv,
                                   t_max=res.cfg.t_max,
                                   exact_inversion=res.cfg.exact_inversion)
            with torch.no_grad():
                y_path = gen.generate(x, gen.prepare(x)).detach()
        finally:
            module.remove()
        m_path = pair(y_path)
        print(f"[{image_id}] 舊下限 VAE 來回 {fmt(m_vae)}", flush=True)
        print(f"[{image_id}] 舊下限 生成路徑 {fmt(m_path)}"
              f"  兩者 LPIPS 差 {abs(m_path['lpips'] - m_vae['lpips']):.2e}",
              flush=True)

        # ---- 2. A1 ----
        # 銳利度帶由這張圖自己的舊下限解出：判準是「不可以比 VAE 自己造成的
        # 更差」（`optimize.recon_floor_thresholds` 同一條線）。
        band = recon.acutance_band(m_vae["acutance_ratio"], args.acut_band)
        target = args.floor_ratio * m_vae[key]
        print(f"[{image_id}] A2 目標 {key}≤{target:.4f}  "
              f"銳利度帶 |1−r|≤{band:.4f}", flush=True)

        # 無約束的 A1 是對照欄，不是備援：它把「壓下限有多少是拿變鈍換來的」
        # 直接放在同一頁上讓人眼判，而那是本專案的主判準（DESIGN §1.1）。
        z0s, h0, s0 = recon.align_latent(
            res.sd, x, perceptual, pair, key=key,
            steps=args.a1_steps, lr=args.a1_lr, w_pixel=w_pixel,
            gamma_acut=0.0, log_every=args.log_every)
        with torch.no_grad():
            y0 = res.sd.decode_latent(z0s)
        m0 = pair(y0)

        z1, h1, s1 = recon.align_latent(
            res.sd, x, perceptual, pair, key=key,
            steps=args.a1_steps, lr=args.a1_lr, w_pixel=w_pixel,
            gamma_acut=args.gamma_acut, band=band,
            log_every=args.log_every)
        with torch.no_grad():
            y1 = res.sd.decode_latent(z1)
        m1 = pair(y1)
        resp1 = recon.latent_response(res.sd, z1, res.suite.pairwise,
                                      seed=args.resp_seed,
                                      scale=args.resp_scale)

        # ---- 3. A2 ----
        with recon.restored(params):
            h2, s2 = recon.finetune_decoder(
                res.sd, x, z1, params, perceptual, pair, key=key,
                steps=args.a2_steps, lr=args.a2_lr, target=target,
                w_pixel=w_pixel, gamma_acut=args.gamma_acut,
                band=band, log_every=max(1, args.log_every // 2))
            with torch.no_grad():
                y2 = res.sd.decode_latent(z1)
            m2 = pair(y2)
            resp2 = recon.latent_response(res.sd, z1, res.suite.pairwise,
                                          seed=args.resp_seed,
                                          scale=args.resp_scale)
            delta = {name: p.detach().cpu().clone() for name, p in tunable}
        torch.save({"z_star": z1.detach().cpu(), "decoder": delta,
                    "image_id": image_id}, out / f"recon_{image_id}.pt")

        if not s2["reached"]:
            print(f"[{image_id}] **A2 未達目標** {key}≤{target:.4f}，"
                  f"最佳 {s2['best']:.4f}（第 {s2['best_step']} 步）。"
                  "如實記錄，該值即為此參數組在本圖上的容量上限", flush=True)

        # ---- 4. 落盤 ----
        save = {
            "orig": x, "floor": y_path, "free": y0, "a1": y1, "a2": y2,
            "floor_diff": diff_map(y_path, x), "free_diff": diff_map(y0, x),
            "a1_diff": diff_map(y1, x), "a2_diff": diff_map(y2, x),
        }
        for name, img in save.items():
            executors.save_image(img, out / f"{image_id}__{name}.png")

        row = {"image_id": image_id, "objective": key,
               "target": round(target, 4), "w_pixel": w_pixel,
               "band": round(band, 4), "gamma_acut": args.gamma_acut,
               "a2_reached": s2["reached"], "a2_stop_step": s2["stop_step"],
               "a1_best_step": s1["best_step"],
               "free_best_step": s0["best_step"],
               "path_minus_vae_lpips": round(m_path["lpips"] - m_vae["lpips"], 6),
               "resp_a1_dists": round(resp1["dists"], 4),
               "resp_a2_dists": round(resp2["dists"], 4),
               "resp_ratio": round(resp2["dists"] / resp1["dists"], 4),
               "n_tunable": n_tunable}
        for stage, m in (("floor", m_path), ("free", m0), ("a1", m1),
                         ("a2", m2)):
            row.update({f"{stage}_{k}": round(m[k], 4) for k in KEYS})
        rows.append(row)
        detail[image_id] = {"free": {"history": h0, "summary": s0},
                            "a1": {"history": h1, "summary": s1},
                            "a2": {"history": h2, "summary": s2},
                            "vae_floor": m_vae, "path_floor": m_path,
                            "resp_a1": resp1, "resp_a2": resp2}
        print(f"[{image_id}] A1 無約束 {fmt(m0)}\n"
              f"[{image_id}] A1 {fmt(m1)}\n[{image_id}] A2 {fmt(m2)}\n"
              f"[{image_id}] latent 反應 DISTS {resp1['dists']:.4f} → "
              f"{resp2['dists']:.4f}（比 {row['resp_ratio']:.3f}）", flush=True)

    executors.write_csv(out / "floor_ab.csv", rows)
    (out / "floor_ab.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                    "rows": rows, "detail": detail},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    page = render([(out.name, out)], out / "compare.html")
    print(f"\n表：{out / 'floor_ab.csv'}\n比對頁：{page}")
    return page


def fmt(m) -> str:
    return (f"LPIPS {m['lpips']:.4f}  DISTS {m['dists']:.4f}  "
            f"PSNR {m['psnr']:.2f} dB  銳利度比 {m['acutance_ratio']:.4f}")


def b64(path: Path) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii"))


CSS = """body{background:#16171b;color:#e8e8ea;font:13px/1.5 system-ui,
sans-serif;margin:24px}h1{font-size:18px}h2{font-size:15px;margin-top:28px}
table{border-collapse:collapse}td,th{border:1px solid #2e3038;padding:6px;
text-align:center;vertical-align:top}th{background:#1e2027}
img{display:block;width:210px;height:210px;object-fit:contain;background:#000}
.m{font:11px ui-monospace,monospace;color:#9aa0aa}.bad{color:#ff8f6b}
.good{color:#7fd18a}.hint{color:#9aa0aa;max-width:64em}"""


def cell(dir_: Path, image_id: str, stage: str, m: dict, diff: str = None
         ) -> str:
    """一格：影像（可再加一張差分）加四項指標。"""
    h = [f"<td><img src='{b64(dir_ / f'{image_id}__{stage}.png')}'>"]
    if diff is not None:
        h.append(f"<img src='{b64(dir_ / f'{image_id}__{diff}.png')}'>")
    h.append(f"<span class=m>LPIPS {m['lpips']}<br>DISTS {m['dists']}<br>"
             f"PSNR {m['psnr']}<br>銳利 {m['acutance_ratio']}</span></td>")
    return "".join(h)


def render(sections, page: Path) -> Path:
    """逐影像一列、逐階段一欄。原圖與舊下限各佔一欄作為錨點。

    影像一律 base64 內嵌：這一頁要能單獨寄出去看（CLAUDE.md 把 `compare.html`
    列為主要產出物，而外連檔案的頁面搬到別的機器就是一片破圖）。
    """
    h = ["<!-- 由 scripts/recon_floor_ab.py 產生 -->",
         "<meta charset='utf-8'>", f"<style>{CSS}</style>",
         "<h1>A 段：φ=0 重建下限（DEC-016）</h1>",
         "<p class=hint>第 2 欄是<strong>現行</strong>生成路徑在 φ=0 時的輸出，"
         "即整條失真預算軸的原點。第 3、4 欄都只換了起點的 latent（A1），"
         "差別在於第 3 欄不設銳利度約束、第 4 欄設；把兩者並排是因為"
         "<strong>壓下限有一部分可以拿變鈍換</strong>，而那要人眼判。"
         "第 5 欄再加上逐圖微調的解碼器（A2）。每張圖下方是它與原圖的差分（×6）"
         "——非加性的失真在原圖上常看不出來，型態要在差分圖上看。"
         "四項指標中 LPIPS 與 DISTS 愈小愈好、PSNR 愈大愈好、"
         "銳利度比愈接近 1 愈好。最右欄的「＊」標的是這一次實際去壓的那一個"
         "下限（`--objective`）；另一個軸列在旁邊，因為 A 段對預算軸"
         "（相對 DISTS，DEC-015）有沒有效益要看的是它。</p>"]
    for label, d in sections:
        d = Path(d)
        rows = list(_read_csv(d / "floor_ab.csv"))
        h.append(f"<h2>{label}</h2>")
        h.append("<table><tr><th>影像</th><th>原圖</th>"
                 "<th>舊下限<br><span class=m>現行路徑</span></th>"
                 "<th>A1 無約束<br><span class=m>只壓 LPIPS</span></th>"
                 "<th>A1<br><span class=m>latent 對齊＋銳利度帶</span></th>"
                 "<th>A1+A2<br><span class=m>再加解碼器微調</span></th>"
                 "<th>相對舊下限</th></tr>")
        for r in rows:
            img = r["image_id"]
            g = lambda stage: {k: r[f"{stage}_{k}"] for k in KEYS}  # noqa: E731
            h.append(f"<tr><td>{img}</td>")
            h.append(f"<td><img src='{b64(d / f'{img}__orig.png')}'></td>")
            h.append(cell(d, img, "floor", g("floor"), "floor_diff"))
            h.append(cell(d, img, "free", g("free"), "free_diff"))
            h.append(cell(d, img, "a1", g("a1"), "a1_diff"))
            h.append(cell(d, img, "a2", g("a2"), "a2_diff"))
            # 兩個感知軸都列。壓下來的是哪一個下限由 --objective 決定，而
            # 另一個是否跟著動，正是 A 段對預算軸（相對 DISTS，DEC-015）
            # 到底有沒有效益的判準，不能只報被優化的那一個。
            rel = ""
            for axis in ("lpips", "dists"):
                f0 = float(r[f"floor_{axis}"])
                star = "＊" if axis == r.get("objective") else ""
                rel += f"{axis.upper()}{star}<br>" + "".join(
                    f"　{name} {100 * (float(r[f'{p}_{axis}']) - f0) / f0:+.1f}%<br>"
                    for name, p in (("無約束", "free"), ("A1", "a1"),
                                    ("A1+A2", "a2")))
            ok = "good" if r["a2_reached"] in ("True", True) else "bad"
            h.append(
                f"<td class=m>{rel}<br>"
                f"<span class={ok}>A2 目標 {r['target']}<br>"
                f"{'達到' if ok == 'good' else '未達'}於第 "
                f"{r['a2_stop_step']} 步</span><br><br>"
                f"銳利度帶 ±{r['band']}<br><br>"
                f"latent 反應<br>DISTS {r['resp_a1_dists']} → "
                f"{r['resp_a2_dists']}<br>比 {r['resp_ratio']}</td>")
            h.append("</tr>")
        h.append("</table>")
        # 探針的能量由該次量測的旗標決定，不在頁面上寫死——寫死的那個值會在
        # 改旗標時無症狀地與圖表分岔。
        scale = float(json.loads(
            (d / "floor_ab.json").read_text(encoding="utf-8"))["args"]
            ["resp_scale"])
        h.append("<p class=hint>「latent 反應」是解碼器對 latent 擾動的反應強度"
                 f"（固定方向、能量 {100 * scale:g}% ‖z*‖）。A2 的風險是解碼器"
                 "把原圖背起來，使 latent 上的擾動傳不到輸出、防禦失去表達管道；"
                 "這個比值掉太多就是那件事發生了，故它與下限一起看。</p>")
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("\n".join(h), encoding="utf-8")
    return page


def _read_csv(path: Path):
    import csv

    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge(dst: Path, srcs) -> Path:
    """疊合逐影像分片。`floor_ab.csv` 串接，其餘檔案直接複製。

    `floor_ab.json` 取第一個分片的：各分片的旗標必須相同（不同就不該疊在
    一起比較），而頁面只從它讀探針的能量設定。逐分片的 `detail` 仍留在各自
    的目錄裡，沒有被丟掉。
    """
    import csv
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in srcs:
        s = Path(s)
        with (s / "floor_ab.csv").open(encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
        for p in s.iterdir():
            if p.suffix in (".png", ".pt"):
                shutil.copy2(p, dst / p.name)
        if not (dst / "floor_ab.json").exists():
            shutil.copy2(s / "floor_ab.json", dst / "floor_ab.json")
    executors.write_csv(dst / "floor_ab.csv", rows)
    print(f"疊合 {len(srcs)} 個分片、{len(rows)} 列 → {dst}")
    return dst


def main() -> None:
    args, rest = build_parser().parse_known_args()
    if args.merge:
        merge(Path(args.merge[0]), args.merge[1:])
        return
    if args.page_only:
        if args.page is None:
            raise SystemExit("--page-only 要一併給 --page")
        sections = []
        for item in args.page_only:
            if "=" not in item:
                raise SystemExit(f"--page-only 的項目要寫成 `標籤=路徑`：{item}")
            label, _, path = item.partition("=")
            sections.append((label, Path(path)))
        print(f"比對頁：{render(sections, args.page)}")
        return
    if args.out is None or not args.images:
        raise SystemExit("量測模式要給 --out 與 --images")
    measure(args, rest)


if __name__ == "__main__":
    main()
