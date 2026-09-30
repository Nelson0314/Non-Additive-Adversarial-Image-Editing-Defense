"""用 VQA 回答「這個指令完成了嗎」，輸出待與人的標註對照。

這一支是**儀器，不是判準**。它只出答案與解析後的判定，一致率算在
`scripts/summarise_vqa.py`，而准不准進讀數由人看那個一致率決定。

為什麼不是 CLIP
────────────────────────────────────────────────────────────────────
`src/defense/criterion.py` 寫明三種 CLIP 寫法都在圖上被推翻過：搜到的點被判成
「指令沒完成」，而防禦後的編輯圖上衣服確實換成指令要的顏色、人也認得出來。
CLIP 在這個粒度上的增益只有 1% 上下，與防禦自己造成的位移同量級。VQA 問的是
一個有具體指涉的問題（「這個人有沒有戴帽子」），答案落在離散的字面上，不靠
一個被雜訊淹沒的連續相似度。

負例是這支腳本的重點
────────────────────────────────────────────────────────────────────
只問編輯圖沒有意義：一個永遠回答 yes 的模型會拿到滿分。所以**原圖也要問**——
三張原圖都沒有戴帽、沒有圍巾，正確答案是 no；衣物顏色的正確答案是原本的顏色。
負例答錯的模型不能用，不管它在正例上多準。

決定性
────────────────────────────────────────────────────────────────────
`do_sample=False` 貪婪解碼。讀數每次跑要給一樣的答案，否則它自己就是一個雜訊源。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'

COLUMNS = ['image', 'class', 'kind', 'role', 'arm', 'seed', 'file', 'question',
           'expect', 'answer_raw', 'verdict']

COLOURS = ('red', 'green', 'blue', 'white', 'black', 'grey', 'gray', 'silver',
           'pink', 'salmon', 'yellow', 'orange', 'purple', 'brown', 'navy',
           'cream', 'beige', 'maroon', 'teal')


def parse_binary(text: str):
    """把自由文字收斂成 yes／no／unclear。開頭的字優先，句中的否定其次。"""
    t = text.strip().lower()
    head = re.split(r'[^a-z]+', t, maxsplit=2)
    head = [h for h in head if h]
    if head and head[0] in ('yes', 'no'):
        return head[0]
    if re.search(r'\bnot? (wearing|a |any )', t) or t.startswith("isn't"):
        return 'no'
    if re.search(r'\bis wearing\b|\bwears\b|\bthere is a\b', t):
        return 'yes'
    return 'unclear'


def parse_colour(text: str):
    """取答案裡第一個出現的顏色詞，沒有就 unclear。"""
    t = text.strip().lower()
    hits = [(t.find(c), c) for c in COLOURS if c in t]
    if not hits:
        return 'unclear'
    c = min(hits)[1]
    return 'grey' if c == 'gray' else c


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--probe', type=Path,
                    help='scripts/instruction_probe.py 的輸出目錄（校準模式）')
    ap.add_argument('--edits', type=Path,
                    help='scripts/evaluate_defence.py 的輸出目錄（防禦圖模式）；'
                         '逐分片目錄都收，未防禦與防禦後的編輯一起問')
    ap.add_argument('--purifier', default='identity',
                    help='防禦圖模式下只問這一道淨化的編輯輸出')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--max-new-tokens', type=int, default=24)
    args = ap.parse_args()

    if bool(args.probe) == bool(args.edits):
        raise SystemExit('--probe 與 --edits 二擇一')
    spec = json.loads(args.config.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)

    def expects(cl, image):
        if cl['name'] == 'garment_colour':
            return cl['expect'][image], ''
        return 'yes', 'no'

    items = []
    if args.probe:
        root = args.probe
        for cl in spec['classes']:
            q = cl['check_question']
            for image in spec['images']:
                orig = root / f'{image}__original.png'
                if not orig.exists():
                    raise SystemExit(f'找不到原圖 {orig}；先跑 instruction_probe.py')
                pos, neg = expects(cl, image)
                items.append({'image': image, 'class': cl['name'],
                              'kind': cl.get('kind', ''), 'role': 'original',
                              'seed': '', 'file': str(orig.relative_to(root)),
                              'question': q, 'expect': neg})
                for seed in spec['seeds']:
                    name = f'{image}__{cl["name"]}__s{seed}__clean_edit.png'
                    if not (root / name).exists():
                        raise SystemExit(f'找不到 {name}')
                    items.append({'image': image, 'class': cl['name'],
                                  'kind': cl.get('kind', ''),
                                  'role': 'clean_edit', 'seed': seed,
                                  'file': name, 'question': q, 'expect': pos})
    else:
        root = args.edits
        found = sorted(root.rglob('*_edit.png'))
        if not found:
            raise SystemExit(f'{root} 下找不到任何 *_edit.png')
        qs = {cl['name']: cl['check_question'] for cl in spec['classes']}
        by_class = {cl['name']: cl for cl in spec['classes']}
        for p in found:
            stem = p.stem
            if stem.endswith('__clean_edit'):
                body, role, arm = stem[:-len('__clean_edit')], 'clean_edit', 'undefended'
            elif stem.endswith('__def_edit'):
                body, role, arm = stem[:-len('__def_edit')], 'def_edit', None
            else:
                continue
            parts = body.split('__')
            if len(parts) < 3:
                continue
            seed = parts[-1].lstrip('s')
            if role == 'def_edit':
                arm = parts[-2]
                pname = parts[-3]
                cls = parts[-4]
                image = '__'.join(parts[:-4])
            else:
                pname = parts[-2]
                cls = parts[-3]
                image = '__'.join(parts[:-3])
            if pname != args.purifier or cls not in qs:
                continue
            pos, neg = expects(by_class[cls], image)
            items.append({'image': image, 'class': cls,
                          'kind': by_class[cls].get('kind', ''),
                          'role': role, 'arm': arm, 'seed': seed,
                          'file': str(p.relative_to(root)).replace('\\', '/'),
                          'question': qs[cls], 'expect': pos})

    neg = sum(1 for i in items if i['role'] == 'original')
    print(f'{args.model}：{len(items)} 題（原圖負例 {neg} 題，'
          f'未防禦編輯 {sum(1 for i in items if i["role"] == "clean_edit")} 題，'
          f'防禦後編輯 {sum(1 for i in items if i["role"] == "def_edit")} 題）',
          flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='auto')
    model.eval()

    rows = []
    for k, it in enumerate(items):
        img = Image.open(root / it['file']).convert('RGB')
        prompt = (f"{it['question']} Answer with one short phrase.")
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors='pt')
        inputs = {k2: v.to(model.device) for k2, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False,
                                 max_new_tokens=args.max_new_tokens)
        gen = out[0][inputs['input_ids'].shape[1]:]
        answer = processor.decode(gen, skip_special_tokens=True).strip()
        verdict = (parse_colour(answer) if it['class'] == 'garment_colour'
                   else parse_binary(answer))
        rows.append({**it, 'answer_raw': answer, 'verdict': verdict})
        print(f"  [{k + 1}/{len(items)}] {it['file'][:52]:52s} "
              f"→ {verdict:8s} | {answer[:40]}", flush=True)

    out_csv = args.out / 'vqa_answers.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {out_csv}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()
