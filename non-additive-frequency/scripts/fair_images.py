"""公平比較三階段共用的影像清單。**一次產生、存檔、之後只讀不重算。**

三份清單是**巢狀**的：25 ⊂ 75 ⊂ 150。校準（25）與抗淨化（75）都是正式表
（150）的子集，否則階段之間的數字無法互相解釋——t0820 那批就是因為地板只
跑到 20 張、與其他條件的交集要事後求，才讓淨增益的比較變得難讀。

分層依 OmniEdit 的五類任務（目錄名前綴），每類等距取樣，不取前 N 張：
dev split 依任務排序，取前 N 會整批落在同一類。
"""
from __future__ import annotations

import argparse
import pathlib

TASKS = ("task_attr_mod", "task_env", "task_obj_add", "task_obj_remove",
         "task_obj_swap")


def strata(root: pathlib.Path) -> dict[str, list[str]]:
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    out: dict[str, list[str]] = {t: [] for t in TASKS}
    for n in names:
        for t in TASKS:
            if n.startswith(t):
                out[t].append(n)
                break
        else:
            raise SystemExit(f"{n} 不屬於已知的五類任務 {TASKS}")
    return out


def pick(root: pathlib.Path) -> dict[int, list[str]]:
    """三份巢狀清單。**由大往小取子集**，不是各自獨立等距取樣。

    各自取樣時 25 張的索引不會落在 75 張的索引上（實測 25 有 5 張不在 75
    裡），巢狀性就破了。故 150 = 全部、75 = 每類每隔一張、25 = 再每隔兩張。
    """
    st = strata(root)
    out: dict[int, list[str]] = {}
    for total, step in ((150, 1), (75, 2), (25, 6)):
        per = total // len(TASKS)
        chosen: list[str] = []
        for t, ns in st.items():
            sub = ns[::step][:per]
            if len(sub) < per:
                raise SystemExit(f"{t} 以 step={step} 只取得 {len(sub)} 張，"
                                 f"不足 {per}")
            chosen += sub
        out[total] = sorted(chosen)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data/omniedit150"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/fair0820"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lists = pick(args.data)
    # 巢狀性：小清單必須整份落在大清單裡，否則階段之間不可比
    for a, b in ((25, 75), (75, 150)):
        miss = set(lists[a]) - set(lists[b])
        if miss:
            raise SystemExit(f"{a} 不是 {b} 的子集，差 {sorted(miss)[:5]}")
    for n, ns in lists.items():
        p = args.out / f"images{n}.txt"
        p.write_text("\n".join(ns) + "\n", encoding="utf-8")
        print(f"{p}：{len(ns)} 張")


if __name__ == "__main__":
    main()
