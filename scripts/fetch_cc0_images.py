"""從 Wikimedia Commons 取 CC0／公有領域影像作為資料集候選。

    python scripts/fetch_cc0_images.py --out data/_raw --per_class 14
    python scripts/fetch_cc0_images.py --out data/_raw --only dog --per_class 30

只收放棄全部著作權的授權（見 `ALLOWED`）：CC0、Public domain、PDM、
No restrictions。**不收 CC BY 與 CC BY-SA**——那些要求姓名標示，在論文附圖
與衍生資料集上會產生持續的義務。授權由 Commons 的 `extmetadata` 欄位判定，
不靠檔名或分類猜測。

為什麼不用 Openverse：其匿名 API 的額度極低，實測連續查詢六類就回 401／429。
Commons 的 API 不需要授權且額度寬鬆。代價是相關性較差——搜 dog 會回傳
繪畫、館藏文物與路牌，故查詢詞帶了一串否定詞，且**下載的是候選池不是
資料集**，仍須人眼挑選。

每類寫一份 `attribution.json`：Commons 檔名、授權、作者、來源頁。CC0 與
公有領域不要求標示，但資料集的可追溯性是本專案的規範（CLAUDE.md 的資料
保全），與是否被要求無關。

挑完之後用 `scripts/prepare_dataset.py` 正規化進 `data/lo_aligned/`。
"""

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia 的 robot policy 要求 User-Agent 帶得到聯絡方式，否則一律 429
# （實測訊息：「Your request does not comply with our robot policy」）。
UA = ("WACV-research/1.0 "
      "(https://github.com/Nelson0314/Non-Additive-Adversarial-Image-Editing-Defense; "
      "getyou318@gmail.com)")

# `incategory:"CC-Zero"` 比全文搜尋準得多。全文搜尋加一串否定詞仍會漂到
# 檔案館藏的歷史肖像——實測搜 cat 回傳的是拳擊手肖像。改用分類過濾後，
# 回傳的是現代照片且授權欄位一致為 CC0。
CC0_CAT = 'incategory:"CC-Zero"'

# 大都會博物館把整批館藏以 CC0 釋出，於是 CC-Zero 分類裡混著大量文物照
# （胸針、雕像、頭骨標本、繪畫翻拍）。這些不是活體動物的自然影像，
# 對編輯攻擊的行為與一般照片不同，必須排除。
ART = "-MET -brooch -ornament -figurine -cranium -skull -sculpture -painting -mummy -vase"

# **人物類不在此表。** 基準論文對人臉的處理是生成而非取用真實照片
# （其註腳：「For ethical considerations, the human face image is synthesized
# by the diffusion model」）。CC0 只放棄著作權，不處理肖像權，而本專案的主題
# 正是對人臉影像的攻擊與保護。man／woman 兩類改由
# `scripts/generate_person_images.py` 以 SD v1.4 生成，與論文一致。
QUERIES = {
    "dog": [f"dog {CC0_CAT} {ART}", f"puppy {CC0_CAT} {ART}"],
    "cat": [f"cat {CC0_CAT} {ART}", f"kitten {CC0_CAT} {ART}"],
    # horse：CC-Zero 裡「horse」大量命中英國酒吧招牌（The White Horse）與
    # 馬具館藏，故排除場所與器物詞。
    "horse": [f"horse {CC0_CAT} {ART} -pub -inn -sign -equipment -harness -qrcode -album",
              f"pony {CC0_CAT} {ART} -pub -inn -sign"],
    # bird：Naturalis Biodiversity Center 把整批標本照以 CC0 釋出，佔滿了
    # 結果。標本是去除背景的死體照，與活體自然影像的統計性質不同。
    "bird": [f"bird {CC0_CAT} {ART} -Naturalis -RMNH -specimen -skeleton -taxidermy -egg",
             f"songbird {CC0_CAT} {ART} -Naturalis -RMNH -specimen"],
}

# extmetadata 的 LicenseShortName 值。只列放棄全部著作權者。
ALLOWED = {"cc0", "public domain", "pdm", "no restrictions", "pd"}


def search(term: str, limit: int, offset: int = 0) -> dict:
    params = {
        "action": "query", "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "gsroffset": str(offset),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": "800",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def _plain(em: dict, key: str) -> str:
    import re

    v = (em.get(key) or {}).get("value") or ""
    return re.sub(r"<[^>]+>", "", str(v)).strip()


def dedup_key(title: str) -> str:
    """同一批上傳的標題只差尾端的序號，正規化後可去重。

    實測搜 cat 回傳九張標題只差編號的同一隻貓。九張近乎相同的影像進資料集，
    會讓「六類各四張」的統計效力遠低於張數看起來的樣子。
    """
    import re

    t = title[5:] if title.startswith("File:") else title
    t = re.sub(r"\.[a-zA-Z]{3,4}$", "", t)
    t = re.sub(r"[\d\-_()]+", " ", t)
    return " ".join(t.lower().split()[:5])


def usable(ii: dict, min_side: int) -> bool:
    em = ii.get("extmetadata", {})
    lic = _plain(em, "LicenseShortName").lower()
    if not any(a in lic for a in ALLOWED):
        return False
    w, h = ii.get("width") or 0, ii.get("height") or 0
    if min(w, h) < min_side:
        return False
    # 極端長寬比在置中裁切後會丟掉大半畫面，主體常被切掉。
    if max(w, h) / max(1, min(w, h)) > 2.0:
        return False
    return bool(ii.get("thumburl") or ii.get("url"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/_raw")
    ap.add_argument("--per_class", type=int, default=14)
    ap.add_argument("--min_side", type=int, default=640)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    classes = [args.only] if args.only else list(QUERIES)
    manifest = {}
    for cls in classes:
        if cls not in QUERIES:
            raise SystemExit(f"未知的類別 {cls!r}；可用的是 {sorted(QUERIES)}")
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        pool, seen = [], set()
        for term in QUERIES[cls]:
            offset = 0
            while len(pool) < args.per_class * 3 and offset < 200:
                try:
                    data = search(term, 40, offset)
                except Exception as e:
                    print(f"  [{cls}] 查詢失敗: {e}")
                    break
                pages = (data.get("query") or {}).get("pages") or {}
                if not pages:
                    break
                for p in pages.values():
                    ii = (p.get("imageinfo") or [{}])[0]
                    k = dedup_key(p["title"])
                    if k in seen or not usable(ii, args.min_side):
                        continue
                    seen.add(k)
                    pool.append((p["title"], ii))
                offset += 40
                time.sleep(1.5)
            if len(pool) >= args.per_class * 3:
                break
        # 打散再取。固定 seed 使同一次查詢的選擇可重現。
        random.Random(20260803).shuffle(pool)
        picked = pool[:args.per_class]

        recs = []
        for i, (title, ii) in enumerate(picked):
            em = ii.get("extmetadata", {})
            src = ii.get("thumburl") or ii["url"]
            path = d / f"cand_{i:02d}.jpg"
            try:
                req = urllib.request.Request(src, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    path.write_bytes(resp.read())
            except Exception as e:
                print(f"  [{cls}] 下載失敗 {title[:40]}: {e}")
                continue
            recs.append({
                "file": path.name,
                "commons_title": title,
                "license": _plain(em, "LicenseShortName"),
                "artist": _plain(em, "Artist"),
                "credit": _plain(em, "Credit")[:200],
                "descriptionurl": ii.get("descriptionurl"),
                "source_url": src,
                "original_size": [ii.get("width"), ii.get("height")],
            })
            print(f"  [{cls}] {path.name}  {recs[-1]['license'][:18]:<18} "
                  f"{title[5:55]}")
            time.sleep(1.0)
        (d / "attribution.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest[cls] = len(recs)

    print("\n候選池：", manifest)
    print(f"下一步：看 {out} 底下的圖挑構圖乾淨的，"
          f"再用 scripts/prepare_dataset.py 正規化")


if __name__ == "__main__":
    main()
