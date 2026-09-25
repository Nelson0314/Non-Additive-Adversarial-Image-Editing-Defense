"""指令完成度（VQA）與編輯後身分：與位移並列的第二組讀數（`docs/NEXT_PLAN.md` A5）。

每格問一個與指令對應的是非題，只問「物件有沒有出現」，不問顏色：顏色類問題在
顏色載體上會被防禦本身滿足。

    p0 Let the person wear sunglasses   → Is the person wearing sunglasses?
    p1 Let the person wear a police suit → Is the person wearing a police uniform?
    p2 Let the person wear a helmet      → Is the person wearing a helmet?
    p3 Let the person wear a bowtie      → Is the person wearing a bow tie?

原圖也問同樣四題當負例（正確答案都是 no）；負例答錯的格子逐列標出，不刪除。
解碼為貪婪（`do_sample=False`），同一張圖每次答案相同。

`id_edit` 為編輯輸出與原圖的 ArcFace 餘弦（`edit_preflight.identity_row` 的同一個
實作）；偵測不到臉時留空，不補 0。

輸出 `lab/results/passthrough/vqa.csv`，每列一格：臂、影像、指令、種子、答案原文、
判定（yes／no／unclear）、`id_edit`。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402

from passthrough_readout import IMAGES, R, defence_png, edit_png, original  # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
QUESTIONS = ["Is the person wearing sunglasses?",
             "Is the person wearing a police uniform?",
             "Is the person wearing a helmet?",
             "Is the person wearing a bow tie?"]
MAIN_SEED = 20260812


def parse_binary(text: str) -> str:
    """與 anti-purification/scripts/instruction_vqa.py 的同名函式相同：開頭的字優先。"""
    t = text.strip().lower()
    head = [h for h in re.split(r"[^a-z]+", t, maxsplit=2) if h]
    if head and head[0] in ("yes", "no"):
        return head[0]
    if re.search(r"\bnot? (wearing|a |any )", t) or t.startswith("isn't"):
        return "no"
    if re.search(r"\bis wearing\b|\bwears\b|\bthere is a\b", t):
        return "yes"
    return "unclear"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["undefended", "ab_warp", "ab_warp_ch",
                                                  "ab_warp_s16", "ab_prism"])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[MAIN_SEED, 20260813, 20260814, 20260815, 20260816])
    ap.add_argument("--out", type=Path, default=R / "lab/results/passthrough/vqa.csv")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VQA 需要 GPU；不要靜默退回 CPU")
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from edit_preflight import identity_row
    from src.utils.io import load_image_tensor

    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    def ask(path, question):
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                 {"type": "text", "text": f"{question} Answer with one short phrase."}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = {k: v.to(model.device) for k, v in
                  processor(text=[text], images=[img], return_tensors="pt").items()}
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    device = torch.device("cuda")
    rows = []
    for name in IMAGES:                         # 負例：原圖
        for k, q in enumerate(QUESTIONS):
            ans = ask(original(name), q)
            rows.append({"arm": "original", "image": name, "prompt_index": k, "seed": "",
                         "question": q, "answer_raw": ans, "verdict": parse_binary(ans),
                         "expect": "no", "id_edit": ""})
    for seed in args.seeds:
        for arm in args.arms:
            for name in IMAGES:
                if arm != "undefended" and not defence_png(arm, name).is_file():
                    continue                   # 隨機對照在該影像沒有合格候選
                x = load_image_tensor(original(name), device, size=512)
                for k, q in enumerate(QUESTIONS):
                    path = edit_png(arm, name, k, None if seed == MAIN_SEED else seed)
                    if not path.is_file():
                        raise SystemExit(f"找不到 {path}")
                    ans = ask(path, q)
                    ident = identity_row(x, load_image_tensor(path, device, size=512))
                    rows.append({"arm": arm, "image": name, "prompt_index": k, "seed": seed,
                                 "question": q, "answer_raw": ans, "verdict": parse_binary(ans),
                                 "expect": "yes", "id_edit": ident["arcface_orig"]})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            yes = sum(r["verdict"] == "yes" for r in rows if r["arm"] == arm and r["seed"] == seed)
            print(f"[vqa] seed {seed} {arm}: yes {yes}/32", flush=True)


if __name__ == "__main__":
    main()
