"""從 Wikimedia Commons 取候選影像池。**只下載，不挑選、不裁切。**

為什麼是 Commons
────────────────────────────────────────────────────────────────────
授權是可查的，且每一張都附得出出處與作者。挑選與裁切分在
`scripts/screen_candidates.py` 與 `scripts/build_dataset.py`，理由是**下載是
不可重現的**（Commons 的內容會變），所以下載這一步要把當下取到的 metadata
逐張留檔，後面兩步才可以純由本地檔案重跑。

品質這一關由 Commons 自己把
────────────────────────────────────────────────────────────────────
搜尋預設加上 `incategory:"Quality images"`——Commons 社群逐張審過的攝影品質
分類（構圖、對焦、曝光、雜訊）。這比自己訂一組門檻可靠，也直接對著「照片
品質要正常」這個要求。`--no-quality-filter` 可以關掉，關掉時池子會大很多，
但要靠 `screen_candidates.py` 的數值欄位自己過濾。

授權
────────────────────────────────────────────────────────────────────
`LICENCE_ALLOW` 是允許的授權短名前綴。CC0 與 Public domain 不要求標示，
CC BY 與 CC BY-SA 要求標示作者——**兩者都逐張把作者與檔案頁寫進
`attribution.json`**，所以差別只在下游怎麼引用，不在能不能用。
不在清單內的（GFDL、Fair use、無授權欄）一律不下載。

用法
    python scripts/fetch_commons_pool.py --out data/_pool
    python scripts/fetch_commons_pool.py --out data/_pool --classes cat dog --per-class 80
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://commons.wikimedia.org/w/api.php"
UA = "image-immunization-research/1.0 (academic research; contact via Wikimedia talk)"
# Commons 對匿名請求限流。429 是**速率**問題不是內容問題，故有界重試＋退避；
# 重試用盡就讓它拋出來，不要吞掉。
RETRIES = 5
BACKOFF = 4.0

# 每類的搜尋詞。目標是**單一、清楚、佔畫面中等比例的主體**，所以詞條偏向
# 「整隻／半身」而不是特寫或群體。
QUERIES = {
    # 人像兩類**不可以用 portrait 當搜尋詞**：Commons 的 Quality images 裡
    # 「portrait」幾乎全是大理石胸像與油畫的翻拍（實測入選的 9 張 man 有 8 張
    # 是雕像或畫框）。改用 `Quality images of people` 這個分類，它收的是人的
    # 照片；雕像與畫作仍會混進來幾張，靠看圖剔掉。
    "man": ["man", "male", "man working", "young man", "older man"],
    "woman": ["woman", "female", "woman working", "young woman", "older woman"],
    "cat": ["domestic cat sitting", "cat outdoors", "felis catus"],
    "dog": ["dog sitting", "dog outdoors", "canis lupus familiaris"],
    "horse": ["horse standing", "horse in field", "equus caballus"],
    "bird": ["bird perched", "bird standing", "passerine"],
}

LICENCE_ALLOW = ("CC0", "Public domain", "CC BY 4.0", "CC BY 3.0", "CC BY 2.0",
                 "CC BY-SA 4.0", "CC BY-SA 3.0", "CC BY-SA 2.0")
MIN_SIDE = 900          # 512 的裁切不得放大，留餘裕給正方形裁切
MAX_ASPECT = 2.2        # 過長的全景裁不出合理的正方形
# 下載縮圖而不是原檔：原檔常是數千萬像素、每張數 MB，而下游只要 512²。
THUMB_WIDTH = 1280


def _fetch(url: str, timeout: int = 60) -> bytes:
    """有界重試的 GET。只對 429 與 5xx 退避重試，其餘立刻拋出。"""
    from urllib.error import HTTPError, URLError

    last = None
    for attempt in range(RETRIES):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=timeout) as r:
                return r.read()
        except HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            last = exc
        except URLError as exc:
            last = exc
        time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"{url} 重試 {RETRIES} 次仍失敗：{last}")


def _get(params: dict) -> dict:
    url = API + "?" + urlencode(dict(params, format="json"))
    return json.loads(_fetch(url).decode("utf-8"))


def _strip(html: str) -> str:
    out, depth = [], 0
    for ch in html or "":
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return " ".join("".join(out).split())


# 逐類的品質分類。人與其他主體在 Commons 分在不同的審核分類下。
QUALITY_CATEGORY = {
    "man": "Quality images of people",
    "woman": "Quality images of people",
}
DEFAULT_CATEGORY = "Quality images"


def search(term: str, limit: int, quality: bool, cls: str = "") -> list:
    q = term + " filetype:bitmap"
    if quality:
        cat = QUALITY_CATEGORY.get(cls, DEFAULT_CATEGORY)
        q += f' incategory:"{cat}"'
    d = _get({"action": "query", "generator": "search", "gsrnamespace": 6,
              "gsrsearch": q, "gsrlimit": limit, "prop": "imageinfo",
              "iiprop": "url|size|extmetadata", "iiurlwidth": THUMB_WIDTH})
    return list(d.get("query", {}).get("pages", {}).values())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/_pool"))
    ap.add_argument("--classes", nargs="+", default=list(QUERIES))
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--no-quality-filter", action="store_true")
    args = ap.parse_args()

    for cls in args.classes:
        d = args.out / cls
        d.mkdir(parents=True, exist_ok=True)
        got, seen = [], set()
        per_term = max(10, args.per_class // len(QUERIES[cls]) + 10)
        for term in QUERIES[cls]:
            time.sleep(1.5)
            for p in search(term, per_term, not args.no_quality_filter, cls):
                if len(got) >= args.per_class:
                    break
                title = p["title"]
                if title in seen:
                    continue
                seen.add(title)
                ii = p["imageinfo"][0]
                em = ii.get("extmetadata", {})
                lic = _strip(em.get("LicenseShortName", {}).get("value", ""))
                if not lic.startswith(LICENCE_ALLOW):
                    continue
                w, h = ii["width"], ii["height"]
                if min(w, h) < MIN_SIDE or max(w, h) / min(w, h) > MAX_ASPECT:
                    continue
                name = f"{len(got):03d}.jpg"
                url = ii.get("thumburl") or ii["url"]
                blob = _fetch(url, timeout=120)
                (d / name).write_bytes(blob)
                got.append({
                    "file": name, "commons_title": title, "license": lic,
                    "artist": _strip(em.get("Artist", {}).get("value", "")),
                    "credit": _strip(em.get("Credit", {}).get("value", "")),
                    "descriptionurl": ii["descriptionurl"],
                    "source_url": ii["url"], "downloaded_url": url,
                    "original_size": [w, h],
                    "search_term": term,
                })
                time.sleep(1.0)
            if len(got) >= args.per_class:
                break
        (d / "attribution.json").write_text(
            json.dumps(got, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{cls}: {len(got)} 張 -> {d}")


if __name__ == "__main__":
    main()
