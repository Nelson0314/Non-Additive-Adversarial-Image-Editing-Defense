"""由 OmniEdit 取 150 張來源影像，作為 DCT-Shield 對齊比較的主線資料集（DEC-030）。

為什麼是 OmniEdit：DCT-Shield（arXiv:2504.17894）Table 1 就是在它上面跑的。
**論文用的那 150 張取不到**——正文只寫「We use 150 samples from the OmniEdit
dataset covering a variety of edit tasks like object addition, removal,
replacement and modifications of attributes and environment」，沒有 seed、
沒有索引清單、沒有釋出 split，官方 repo 又是空的。故我們能對齊的是**來源與
任務分布**，不是同一批影像；n=150 的抽樣誤差留在差異裡，必須寫進 limitation。

`src_img` 是**真實照片**（OmniEdit 的來源取自 LAION-5B 與 OpenImagesV6，
最低 1 MP），生成的是 `edited_img`，本專案不使用後者。授權：資料集卡標 MIT，
但那是彙編的授權而非底層照片的權利狀態——使用者 2026-08-19 明示撤銷本專案
先前「只收 CC0」的規定（DEC-030），此處照辦並如實記錄。

任務分層
────────────────────────────────────────────────────────────────────
dev split 共 700 列，實測分布為

    swap 200、style 100、attribute_modification 100、env 100、removal 100、addition 100

論文那句涵蓋的正好是 **style 以外的五類**，故每類取 30 張 = 150。
**不可以取前 150 列**——dev split 依任務排序，前面整批是 style。

取得方式
────────────────────────────────────────────────────────────────────
走 HF 的 datasets-server `/rows`，**不必下載 2.85 TB 的完整資料集**。
`/filter` 端點對此資料集回 500，故改為分頁讀完 700 列再自行篩選。

抽樣的可重現性
────────────────────────────────────────────────────────────────────
先依 `omni_edit_id` 排序再以固定 seed 抽樣，並把抽中的 id 全部寫進
`provenance.json`。**論文沒做這件事，所以他們的 150 張永遠取不回來；
我們的要能取回來。**

prompt 的形態差異（重要）
────────────────────────────────────────────────────────────────────
`edited_prompt_list` 是**指令式**的（"Add a hat to the cat"），因為 OmniEdit
與 DCT-Shield 的攻擊模型是 InstructPix2Pix。本專案主線的攻擊是 **SDEdit**，
吃的是**描述式** caption（"a cat wearing a hat"）。兩者不等價。

本腳本**原樣存下指令，不自行改寫成描述**——改寫等於捏造資料。要跑 SDEdit
主線時如何處理，是還沒定的事項，見 `docs/reference/BASELINE_ALIGNMENT.md` §3.3。

版面
────────────────────────────────────────────────────────────────────
每個樣本一個子目錄、目錄裡一張 PNG，於是 `apa_baseline.load_dataset` 不必改
就能讀（它的版面是「每類一子目錄」）。代價是 150 個目錄。

`content`（類別名詞）OmniEdit 沒有提供，一律留空並在 prompts.yaml 註明。
用得到它的只有 `apa_weak` 的階段一 LoRA，而加性 baseline 本輪已擱置（DEC-025）。

用法：
    python scripts/fetch_omniedit.py --out data/omniedit150
    python scripts/fetch_omniedit.py --out /tmp/probe --per-task 2 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "TIGER-Lab/OmniEdit-Filtered-1.2M"
SPLIT = "dev"
PAGE = 100
DEV_ROWS = 700

# 論文那句涵蓋的五類，**不含 style**。
TASKS = ("addition", "removal", "swap", "attribute_modification", "env")
PER_TASK = 30                     # 5 × 30 = 150，與論文的張數一致
RESOLUTION = 512

UA = ("WACV-research/1.0 (https://github.com/Nelson0314/"
      "Non-Additive-Adversarial-Image-Editing-Defense)")


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_rows() -> list:
    """分頁讀完 dev split 的全部列（只有中繼資料與影像 URL，不含影像本體）。"""
    out = []
    for off in range(0, DEV_ROWS, PAGE):
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset": DATASET, "config": "default", "split": SPLIT,
             "offset": off, "length": PAGE})
        d = json.loads(_get(url))
        out.extend(x["row"] for x in d["rows"])
        print(f"  列出 {off}–{off + len(d['rows'])}", flush=True)
        time.sleep(0.5)
    return out


def stratified(rows: list, per_task: int, seed: int) -> list:
    """依任務分層抽樣。先依 `omni_edit_id` 排序才抽，故同一個 seed 必得同一批。"""
    rng = random.Random(seed)
    picked = []
    for task in TASKS:
        pool = sorted((r for r in rows if r["task"] == task),
                      key=lambda r: r["omni_edit_id"])
        if len(pool) < per_task:
            raise SystemExit(
                f"任務 {task} 只有 {len(pool)} 列，抽不出 {per_task} 張。"
                f"dev split 的實測分布是 addition/removal/attribute_modification/env "
                f"各 100、swap 200")
        picked.extend(rng.sample(pool, per_task))
    return picked


def normalize(raw: bytes, size: int):
    """中央正方裁切後縮到 `size`。與 `prepare_dataset.py` 同一套規則。"""
    from PIL import Image

    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    side = min(w, h)
    im = im.crop(((w - side) // 2, (h - side) // 2,
                  (w - side) // 2 + side, (h - side) // 2 + side))
    return im.resize((size, size), Image.LANCZOS), (w, h), side


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=PER_TASK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=RESOLUTION)
    ap.add_argument("--dry-run", action="store_true",
                    help="只列出抽中的 id 與任務，不下載影像")
    args = ap.parse_args()

    print(f"列出 {DATASET} 的 {SPLIT} split（{DEV_ROWS} 列）…", flush=True)
    rows = list_rows()
    picked = stratified(rows, args.per_task, args.seed)
    print(f"\n抽中 {len(picked)} 列（seed={args.seed}，每類 {args.per_task}）")
    if args.dry_run:
        for r in picked:
            print(f"  {r['task']:24s} {r['omni_edit_id']}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    prov, prompts = [], {}
    for i, r in enumerate(picked):
        name = r["omni_edit_id"]
        src = r["src_img"]
        url = src["src"] if isinstance(src, dict) else src
        raw = _get(url)
        im, orig, side = normalize(raw, args.size)
        (args.out / name).mkdir(exist_ok=True)
        im.save(args.out / name / f"{name}.png")
        instr = r.get("edited_prompt_list") or []
        prov.append({
            "output": f"{name}/{name}.png", "omni_edit_id": name,
            "task": r["task"], "source_url": url,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_size": list(orig), "crop_side": side,
            "output_size": args.size,
            "sc_score_1": r.get("sc_score_1"), "pq_score": r.get("pq_score"),
        })
        prompts[name] = {
            "content": "",                    # OmniEdit 未提供類別名詞
            "task": r["task"],
            # **指令式**，不是描述式。原樣存下，不改寫。
            "prompts": list(instr) or [""],
        }
        print(f"  [{i + 1:3d}/{len(picked)}] {r['task']:24s} {name} "
              f"{orig[0]}×{orig[1]} → {args.size}²", flush=True)
        time.sleep(0.3)

    (args.out / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    header = (
        "# OmniEdit 抽樣（DEC-030）。scripts/fetch_omniedit.py 產生，勿手改。\n"
        f"# dataset={DATASET} split={SPLIT} seed={args.seed} "
        f"per_task={args.per_task}\n"
        "#\n"
        "# prompts 是 OmniEdit 的 **指令式** 編輯指令（給 InstructPix2Pix 用），\n"
        "# 不是 SDEdit 吃的描述式 caption。兩者不等價，見腳本 docstring。\n"
        "# content 一律留空——OmniEdit 沒有提供類別名詞，用得到它的只有 apa_weak。\n")
    import yaml

    (args.out / "prompts.yaml").write_text(
        header + yaml.safe_dump(prompts, allow_unicode=True, sort_keys=True),
        encoding="utf-8")
    print(f"\n寫出 {args.out}：{len(prov)} 張、provenance.json、prompts.yaml")


if __name__ == "__main__":
    main()
