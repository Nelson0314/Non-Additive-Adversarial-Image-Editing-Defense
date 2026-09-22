"""未防禦的編輯預檢：確認這批影像**編輯得起來**，再談防禦。

為什麼一定要有這一步
────────────────────────────────────────────────────────────────────
防禦的讀數全部是「防禦後的編輯」對「未防禦的編輯」的比較。未防禦那一側
如果根本沒照 prompt 動、或動到把人整個換掉，分母就不成立，後面每一欄都是
雜訊。本專案量過兩次：strength 0.55 下九張影像沒有一張服從 prompt；
strength 0.8 下服從了，但 FaceNet 對原圖的相似度掉到 0.115（同一人的參照
門檻是 0.55），即**人被換掉**。換資料集就要重跑這一支。

統一的編輯設定
────────────────────────────────────────────────────────────────────
兩個場景共用**同一顆種子、同樣的步數、guidance 與解析度**，差別只有
路徑本身：

| | ip2p（指令式編輯） | inpainting |
|---|---|---|
| 權重 | `timbrooks/instruct-pix2pix` | `runwayml/stable-diffusion-inpainting` |
| 呼叫 | `IP2PWrapper.edit` | `SDWrapper.inpaint` |
| 作用範圍 | 整張影像 | 只有遮罩內 |
| 遮罩 | 無 | `data/<資料集>/masks/`，白 = 重繪 |
| 指令型態 | **加配件**（加了東西、人不變） | **不衝著人**（換背景或在旁邊加東西） |

指令由 `prompts.yaml` 的 `edits` 逐場景給，不逐類別——兩個場景的威脅模型
不同，能問的問題也不同，理由寫在該檔。

為什麼 ip2p 不是 SD img2img
────────────────────────────────────────────────────────────────────
SD v1.4 的 img2img 實測在 strength 0.6 與 0.8 上都把人換掉（FaceNet 對原圖
的中位數 0.120 與 0.029，同一人門檻 0.55，16 格全滅），身分那一欄失去分母。
IP2P 在本專案先前的批次上 `id_orig` 是 0.943，兩件事能兼顧。

四個讀數
────────────────────────────────────────────────────────────────────
| 欄 | 意義 | 怎麼讀 |
|---|---|---|
| `id_orig` | FaceNet（編輯結果 vs 原圖） | **> 0.55 才算同一個人**。低於它代表編輯把人換掉了，身分那一欄失去分母 |
| `arcface_orig` | ArcFace 餘弦 | 第二個身分讀數，與 FaceNet 不同族 |
| `clip_gain` | CLIP(編輯, prompt) − CLIP(原圖, prompt) | 指令有沒有被執行。**本專案量過它會與看圖相反**，只當解釋不當判準 |
| `lpips` / `psnr` | 編輯結果對原圖 | 改動幅度 |

**判準不由本腳本下。** 它只把數與圖擺出來。

用法（遠端，需要一張卡）
    HF_HOME=/var/cache/huggingface CUDA_VISIBLE_DEVICES=<卡> \\
        python code/edit_preflight.py --out images/edit_preflight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402
import yaml  # noqa: E402

from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.ip2p import (  # noqa: E402
    IP2P_SEED, IP2P_STEPS, IP2P_TEXT_GUIDANCE,
    MODEL_NAME as IP2P_MODEL, IP2PWrapper,
)
from src.models.sd import SDInpaintWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
RESOLUTION = 512
EDIT_SEED = IP2P_SEED         # 20260812，兩條路徑共用同一顆
# **兩條路徑統一 50 步**：FaceLock（Table A2）與 DiffusionGuard（Appendix D）
# 都用 50，對齊之後這一批的數字才對得回那兩篇的協定。IP2P 的封裝預設是 100，
# 本腳本明給 50 覆寫掉。
EDIT_STEPS = 50
INPAINT_GUIDANCE = 7.5
# **image guidance 用 1.8，不是 IP2P 封裝預設的 1.5。** 1.5 在 `a helmet`
# 這一句上八張裡有三張多出第二、第三個戴帽的人、四張把主體的臉換掉；1.8 在
# s_t 7.5／6.0／5.0 三個值上都把這兩件事歸零，2.2 則開始畫不出帽體（八張只剩
# 二到四張有帽）。逐格的影像與判讀在 `../anti-purification/runs/ip2p_helmet_sweep/`。四條 ip2p
# 指令共用這個值，改了就四條一起重跑。
IP2P_EDIT_IMAGE_GUIDANCE = 1.8


def load_items(root: Path):
    """回傳 (影像清單, 逐場景的指令表)。指令不逐類別，見 prompts.yaml。"""
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    edits = spec.get("edits")
    if not edits:
        raise SystemExit(
            f"{root / 'prompts.yaml'} 缺 `edits`：指令要按場景分組"
            "（ip2p 加配件、inpaint 不衝著人），舊的逐類別 `prompts` 格式"
            "**不適用**，不要靜默沿用")
    out = []
    for cls in sorted(k for k in spec if k != "edits"):
        for img in sorted((root / cls).glob("*.png")):
            mask = root / "masks" / f"{img.stem}.png"
            out.append({
                "name": img.stem, "class": cls, "path": img,
                "content": spec[cls]["content"],
                "mask": mask if mask.exists() else None,
            })
    return out, {k: list(v) for k, v in edits.items()}


def defended_image(directory: Path, name: str) -> Path:
    """`defence_run.py` 寫出來的是 `<名稱>__<條件>__def.png`，條件名在中間。

    兩種寫法都收（有些條件沒有中段），但**同一張只能對到一個檔案**：對到兩個
    就是目錄裡混了兩個條件的產物，那時候拿哪一張都是錯的。
    """
    matches = sorted(set(directory.glob(f"{name}__*__def.png"))
                     | set(directory.glob(f"{name}__def.png")))
    if len(matches) != 1:
        raise SystemExit(
            f"{directory} 裡 {name} 對到 {len(matches)} 個防禦圖"
            f"（{' '.join(p.name for p in matches) or '沒有'}）：要剛好一個")
    return matches[0]


def apply_defended(items, directory: Path):
    """把輸入影像換成 `directory` 裡的防禦圖，其餘欄位一律不動。

    換的只有「輸入的那張影像」。遮罩是防禦方依 `content` 畫的、指令是攻擊方
    寫的，兩者都不隨防禦條件改變，沿用資料集那一份才是同一條管線。

    **缺圖拋錯而不是略過**：少跑幾張不會有任何症狀，但那一條件的分母就跟
    其他條件不一樣了，後面每個比較都被污染。
    """
    resolved = [defended_image(directory, item["name"]) for item in items]
    for item, source in zip(items, resolved):
        item["path"] = source
    return items


# 身分量測**暫停**（使用者裁決）。理由是這一階段要確認的是「指令有沒有被
# 執行」，而身分那一欄在配件類指令上會被配件本身干擾（口罩／面罩遮住臉時
# FaceNet 抓不到人，掉下去分不出是編輯成功還是防禦生效）。
# 恢復時把 `--identity` 打開即可，函式保留不刪。
def identity_row(x_ref, x_edit) -> dict:
    """兩個身分讀數。**偵測不到臉時留空而不是填 0**——「看不出是誰」與
    「相似度為零」是兩件事，補 0 會讓前者被讀成後者。"""
    from src.metrics import arcface
    from src.metrics.identity import embed, similarity

    e_ref, e_edit = embed(x_ref), embed(x_edit)
    fn = similarity(e_ref, e_edit)
    a_ref, a_edit = arcface.embed(x_ref), arcface.embed(x_edit)
    af = arcface.similarity(a_ref, a_edit)
    return {
        "id_orig": "" if fn is None else round(fn, 4),
        "arcface_orig": "" if af is None else round(af, 4),
        "face_found_orig": e_ref is not None,
        "face_found_edit": e_edit is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--out", type=Path, default=paths.IMAGES / "edit_preflight")
    ap.add_argument("--scenarios", nargs="+", default=["ip2p", "inpaint"])
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--identity", action="store_true",
                    help="加量身分讀數。預設關閉，見 identity_row 的註解")
    ap.add_argument("--metrics-only", action="store_true",
                    help="不重跑編輯，只讀既有的 PNG 重算讀數")
    ap.add_argument("--sampler", choices=("wrapper", "official"),
                    default="official",
                    help="inpaint 走哪一條取樣鏈。`official` 直接呼叫 diffusers "
                         "的 StableDiffusionInpaintPipeline；`wrapper` 走本專案"
                         "可微分的 SDWrapper.inpaint（手寫 DDIM）")
    ap.add_argument("--instruction-set", default=None,
                    help="改用 prompts.yaml 的 edits 底下另一個鍵當指令來源，"
                         "預設用與場景同名的那個鍵")
    ap.add_argument("--suffix", default="",
                    help="輸出子目錄的後綴，用來讓不同的臂並存不互相覆蓋")
    ap.add_argument("--s-t", type=float, default=IP2P_TEXT_GUIDANCE,
                    help="IP2P 的文字 guidance。逐列記進 CSV")
    ap.add_argument("--s-i", type=float, default=IP2P_EDIT_IMAGE_GUIDANCE,
                    help="IP2P 的影像 guidance。調高會保留更多原圖")
    ap.add_argument("--defended", type=Path, default=None,
                    help="改用這個目錄裡的 `<名稱>__def.png` 當輸入，指令、"
                         "遮罩、種子、步數與 guidance 全部不變。"
                         "缺任何一張就拋錯，不靜默略過")
    ap.add_argument("--negative-prompt", default="",
                    help="IP2P negative prompt; recorded verbatim in CSV")
    ap.add_argument("--prompt-indices", nargs="+", type=int, default=None,
                    help="IP2P prompt indices to run, preserving their pN filenames")
    ap.add_argument("--ip2p-instruction", default=None,
                    help="Override exactly one selected IP2P instruction")
    ap.add_argument("--seed", type=int, default=None,
                    help="IP2P seed override; default is 20260812")
    ap.add_argument("--require-new-arm", action="store_true",
                    help="Refuse an existing arm directory or CSV key (for sweeps)")
    args = ap.parse_args()

    ip2p_options = (args.negative_prompt or args.prompt_indices is not None
                   or args.ip2p_instruction is not None or args.seed is not None)
    if ip2p_options and args.scenarios != ["ip2p"]:
        ap.error("IP2P-only options require --scenarios ip2p")
    if args.ip2p_instruction is not None:
        if not args.ip2p_instruction.strip():
            ap.error("--ip2p-instruction must not be empty")
        if args.prompt_indices is None or len(args.prompt_indices) != 1:
            ap.error("--ip2p-instruction requires exactly one --prompt-indices value")
    if args.require_new_arm and args.metrics_only:
        ap.error("--require-new-arm cannot be used with --metrics-only")
    if args.defended is not None and not args.defended.is_dir():
        ap.error(f"--defended 指到的不是目錄：{args.defended}")

    args.out.mkdir(parents=True, exist_ok=True)
    items, edits = load_items(args.data)
    if args.defended is not None:
        apply_defended(items, args.defended)
    if args.images:
        keep = set(args.images)
        items = [d for d in items if d["name"] in keep]
    if not items:
        raise SystemExit("沒有符合的影像")
    if args.instruction_set:
        if args.instruction_set not in edits:
            raise SystemExit(
                f"edits 底下沒有 `{args.instruction_set}`，有的是 {sorted(edits)}")
        for s_ in args.scenarios:
            edits[s_] = edits[args.instruction_set]
    missing = [s for s in args.scenarios if s not in edits]
    if missing:
        raise SystemExit(f"prompts.yaml 的 edits 缺這些場景的指令：{missing}")
    if args.prompt_indices is not None:
        if (len(set(args.prompt_indices)) != len(args.prompt_indices)
                or any(i < 0 or i >= len(edits["ip2p"])
                       for i in args.prompt_indices)):
            ap.error("--prompt-indices must be distinct valid IP2P indices")
        if args.ip2p_instruction is not None:
            edits["ip2p"][args.prompt_indices[0]] = args.ip2p_instruction
    print(f"{len(items)} 張影像，場景 {args.scenarios}")

    # **既有的列要保留。** 只重跑一個場景時，若直接用新的 rows 覆寫 CSV，
    # 另一個場景那 32 列會被洗掉而**沒有任何症狀**（檔案還在、格式也對）。
    # 故以 (場景, 影像, 指令序號) 為鍵合併。
    # 合併的鍵是 **arm**（場景＋後綴），不是場景：同一個場景可以有好幾個臂
    # （不同取樣鏈、不同遮罩、不同指令組），只看場景會讓後跑的臂把先跑的洗掉，
    # 而檔案看起來完整、格式也對，**沒有任何症狀**。
    csv_path = args.out / "preflight.csv"
    keep_rows = []
    touched = {s_ + args.suffix for s_ in args.scenarios}
    if args.require_new_arm:
        existing_dirs = [str(args.out / arm) for arm in touched
                         if (args.out / arm).exists()]
        if existing_dirs:
            ap.error(f"Arm already exists; choose a new --suffix: {existing_dirs}")
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, encoding="utf-8") as f:
            existing_rows = list(_csv.DictReader(f))
        if args.require_new_arm and any(
                r.get("arm", r["scenario"]) in touched for r in existing_rows):
            ap.error("Arm already exists in CSV; choose a new --suffix")
        keep_rows = [r for r in existing_rows
                         if r.get("arm", r["scenario"]) not in touched]
        if keep_rows:
            print(f"沿用 {len(keep_rows)} 列既有讀數"
                  f"（場景 {sorted({r['scenario'] for r in keep_rows})}）")

    rows = []
    for scenario in args.scenarios:
        instructions = edits[scenario]
        d = args.out / (scenario + args.suffix)
        d.mkdir(parents=True, exist_ok=True)

        model = IP2P_MODEL if scenario == "ip2p" else INPAINT_MODEL
        guidance = (args.s_t if scenario == "ip2p" else INPAINT_GUIDANCE)
        seed = (args.seed if scenario == "ip2p" and args.seed is not None
                else EDIT_SEED)
        steps = EDIT_STEPS
        if args.metrics_only:
            # 不載受害模型：這個模式只讀既有的 PNG 重算讀數，載權重是白花
            # 幾分鐘與十幾 GB 顯存。
            victim = None
            device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
        elif scenario == "ip2p":
            victim = IP2PWrapper(dtype=torch.float32)
            device = victim.device
        else:
            # **inpainting 權重要用 SDInpaintWrapper**：它載的是
            # `StableDiffusionInpaintPipeline`，官方臂直接用 `victim.pipe`。
            victim = SDInpaintWrapper(model, dtype=torch.float32)
            device = victim.device
        suite = MetricSuite(device=device)
        print(f"\n=== {scenario}（{model}）===", flush=True)

        for item in items:
            x01 = load_image_tensor(item["path"], device, size=RESOLUTION)
            save_image(x01, d / f"{item['name']}__orig.png")
            mask = None
            if scenario == "inpaint":
                if item["mask"] is None:
                    raise SystemExit(f"{item['name']} 缺遮罩，inpainting 跑不了")
                mask = load_image_tensor(item["mask"], device,
                                         size=RESOLUTION)[:, :1]
                mask = (mask >= 0.5).to(x01.dtype)

            for pi, template in enumerate(instructions):
                if args.prompt_indices is not None and pi not in args.prompt_indices:
                    continue
                # inpaint 的句子要把主體寫進去，否則模型會照上下文補出
                # 另一個人（實測 woman_01 的「狗」被畫成人臉）。
                prompt = template.format(content=item["content"])
                out_png = d / f"{item['name']}__p{pi}.png"

                if args.metrics_only:
                    if not out_png.exists():
                        raise SystemExit(
                            f"--metrics-only 需要既有的 {out_png}，但它不存在")
                    edit = load_image_tensor(out_png, device, size=RESOLUTION)
                elif scenario == "ip2p":
                    with torch.no_grad():
                        edit = victim.edit(x01, prompt, seed=seed,
                                           steps=steps, s_t=args.s_t,
                                           s_i=args.s_i,
                                           negative_prompt=args.negative_prompt or None)
                elif args.sampler == "official":
                    # 官方 pipeline，不經過本專案手寫的 DDIM。用來把
                    # 「取樣鏈寫錯」與「遮罩／畫布不夠」兩件事分開。
                    with torch.no_grad():
                        gen = torch.Generator(device=device).manual_seed(
                            int(EDIT_SEED))
                        raw = victim.pipe(
                            prompt=prompt, image=x01, mask_image=mask,
                            strength=1.0, num_inference_steps=steps,
                            guidance_scale=guidance, generator=gen,
                            output_type="pt").images
                    # **兩張都存**：`raw` 是 pipeline 的原始輸出，`edit` 是
                    # 合成回遮罩外原圖之後的。分開存才分得出「生成本身失敗」
                    # 與「只是合成邊界難看」。
                    if not args.metrics_only:
                        save_image(raw, d / f"{item['name']}__p{pi}__raw.png")
                    edit = mask * raw + (1.0 - mask) * x01
                else:
                    with torch.no_grad():
                        emb = victim.encode_text(prompt)
                        emb_u = victim.uncond_prompt()
                        noise = victim.sample_edit_noise(
                            victim.encode_image(x01), seed=EDIT_SEED)
                        # **不要包 `conditioning_for`**：`inpaint` 自己拼
                        # 9 通道輸入，而 `_latent_in` 的守衛看到 9 通道就直接
                        # 放行，`_inpaint_cond` 在這條路上根本不會被取用，
                        # 包了只是多編碼一次 masked image。
                        edit = victim.inpaint(
                            x01, mask, emb, noise, steps,
                            guidance_scale=guidance, emb_uncond=emb_u)
                        # 遮罩外逐位元保留原圖，**解碼之後合成一次**。
                        # 取樣迴圈裡不做這件事（見 SDWrapper.inpaint 的
                        # docstring 第 3 點）。
                        edit = mask * edit + (1.0 - mask) * x01

                tag = f"p{pi}"
                if not args.metrics_only:
                    save_image(edit, out_png)

                pair = suite.pairwise(x01, edit)
                sem_e = suite.semantic(edit, prompt)
                sem_o = suite.semantic(x01, prompt)
                row = {
                    "scenario": scenario, "image": item["name"],
                    "class": item["class"], "prompt_index": pi,
                    "prompt": prompt, "victim": model,
                    "seed": seed, "steps": steps, "guidance": guidance,
                    "arm": scenario + args.suffix,
                    "instruction_set": args.instruction_set or scenario,
                    "data": str(args.data).replace("\\", "/"),
                    "defence": args.defended.name if args.defended else "",
                    "input_png": item["path"].as_posix(),
                    "sampler": (args.sampler if scenario == "inpaint"
                                else "ip2p_pipeline"),
                    "s_t": args.s_t if scenario == "ip2p" else "",
                    "s_i": args.s_i if scenario == "ip2p" else "",
                    "instruction": prompt,
                    "negative_prompt": (args.negative_prompt
                                        if scenario == "ip2p" else ""),
                    "output_png": out_png.as_posix(),
                    "clip_edit": round(sem_e["clip"], 4),
                    "clip_orig": round(sem_o["clip"], 4),
                    "clip_gain": round(sem_e["clip"] - sem_o["clip"], 4),
                    "lpips": round(float(pair["lpips"]), 4),
                    "psnr": round(float(pair["psnr"]), 3),
                    **(identity_row(x01, edit) if args.identity else {}),
                }
                rows.append(row)
                print(f"  {item['name']} {tag}: clip_gain "
                      f"{row['clip_gain']:+.4f} lpips {row['lpips']}"
                      + (f" id {row['id_orig']}" if args.identity else ""),
                      flush=True)
                write_csv(csv_path, keep_rows + rows)
        del victim, suite  # metrics-only 時 victim 是 None，del 仍合法
        torch.cuda.empty_cache()

    print(f"\n{len(rows)} 格 -> {args.out / 'preflight.csv'}")
    if args.identity:
        same = [r for r in rows
                if r["id_orig"] != "" and float(r["id_orig"]) >= 0.55]
        print(f"FaceNet ≥ 0.55（仍是同一個人）的格數：{len(same)} / {len(rows)}")
    else:
        print("身分讀數本輪未量（--identity 可開啟）")


if __name__ == "__main__":
    main()
