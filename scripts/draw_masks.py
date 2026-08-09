"""人工繪製 inpainting 的攻擊遮罩。滑鼠拖曳，逐影像存成一張 PNG。

為什麼是人工
──────────────────────────────────────────────────────────────────────
遮罩是**攻擊方的設定**——他要重畫哪一塊。文獻裡這一項幾乎都是人工的：
PIE-Bench 附標註遮罩、PhotoGuard 與 AdvPaint 的 inpainting 實驗用人工
遮罩、Lo et al. (CVPR 2024) Figure 3 那張也是手畫的；真實的 inpainting
軟體本來就是讓使用者自己框。本專案先前兩次改由模型的 cross-attention 產
遮罩，兩次都引進了與研究無關的失效（DEF-011，以及「要新增的物件在原圖裡
不存在、注意力落在最像它的東西上」），故改回人工（DEC-010）。

畫的時候要守的一條
──────────────────────────────────────────────────────────────────────
**c_a 要落在遮罩外。** 這是 Lo Figure 3 的配置：遮罩內是攻擊方要重畫的
區域，而要保住的那個物件在遮罩外。畫到主體身上的話，式 (4) 的 M 會落在
會被整片覆寫的區域，防禦擾動一步都活不過——那正是 ip1／ip2／ip3 三批
量不到東西的原因。本工具把 c_a 顯示在標題列提醒；真正的攔截在
`src/defense/optimize.py`，它以 `assert_masks_disjoint` 對真正進損失的 M
斷言不相交，重疊即拋出。

用法
──────────────────────────────────────────────────────────────────────
    python scripts/draw_masks.py                      # 全部 24 張
    python scripts/draw_masks.py --images horse_00 horse_03
    python scripts/draw_masks.py --overview          # 只產總覽圖
    python scripts/draw_masks.py --invert            # 主體 → 主體之外
    python scripts/draw_masks.py --recrop            # 裁切放大到合適占比

遮罩存在 `data/lo_masks/`、裁切前的原圖在 `data/lo_original/`。兩者都
**不放在 `data/lo_aligned/` 裡**：`load_lo_aligned` 拒絕未宣告卻含有 PNG
的子目錄，那道檢查擋的是「忘了宣告類別」，不該為了放遮罩而鬆綁。

遮罩要**貼合物件輪廓**，不是方框。方框會把物件周圍的背景一起劃進重畫區，
於是涵蓋率不再代表「攻擊方改掉了那個物件」，而防禦效果的分母跟著失真。
故本工具提供兩種描邊方式，都在放開／封閉時填成實心區域：

| 動作 | 操作 |
|---|---|
| 套索：沿輪廓拖曳一圈 | 左鍵拖曳（放開即封閉並填滿）|
| 多邊形：逐點點出輪廓 | `V` 切到多邊形，左鍵逐點，Enter 或雙擊封閉 |
| 筆刷修邊 | 右鍵拖曳 |
| 擦除 | 按住 Shift（三種都適用）|
| 筆刷大小 | `[` 縮小、`]` 放大 |
| 復原 | Ctrl+Z |
| 清空本張 | C |
| 存檔並下一張 | N（描邊中的 Enter 是封閉多邊形）|
| 上一張 | P |
| 取消這一筆／離開 | Esc（描邊中取消該筆，否則離開）|

有機的外形用套索快，直邊的物件用多邊形準；兩者可以混用，筆刷補漏。

遮罩以 8-bit 灰階存出，**255 表示要重畫的區域**，與 `SDWrapper.inpaint`
及 diffusers 同一約定。尺寸等於原圖，`run_stage` 載入時再以 NEAREST 縮到
`--resolution`。已存在的遮罩會在開啟時載入，可繼續修改。

涵蓋率顯示在標題列。實測可用窗口是 [0.15, 0.45]：偏小則攻擊方能改的太少、
未防禦的那一側本身就不成立；偏大則物件佔滿畫面（人像 8 張的 c_a 區實測
0.92–1.00，整組不可用）。**本工具只顯示不阻擋**——合格的區間隨物件在畫面
中的佔比而變，一個全域常數正是本專案重複踩過的缺陷型態。
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageTk                    # noqa: E402

import yaml                                                  # noqa: E402

# 可用的涵蓋率窗口。顯示用，不阻擋。
COVERAGE_WINDOW = (0.15, 0.45)
# 遮罩疊在原圖上的顏色與不透明度。紅色與資料集裡的動物／人物都拉得開。
OVERLAY_RGB = (255, 40, 40)
OVERLAY_ALPHA = 0.45


def load_targets(root: Path, ids: Optional[List[str]]) -> List[Tuple[str, str, Path]]:
    """回傳 [(image_id, c_a, png 路徑)]，順序與 `load_lo_aligned` 一致。

    直接讀 `prompts.yaml` 而不經 `executors.load_lo_aligned`：後者會把影像
    載成 tensor 並搬上 device，本工具只要路徑與 c_a，不需要 torch。
    """
    pf = root / "prompts.yaml"
    if not pf.exists():
        raise SystemExit(f"{pf} 不存在")
    spec = yaml.safe_load(pf.read_text(encoding="utf-8")) or {}

    out: List[Tuple[str, str, Path]] = []
    for cls in sorted(spec):
        d = root / cls
        if not d.is_dir():
            raise SystemExit(f"{pf} 宣告了類別 {cls!r}，但 {d} 不存在")
        for p in sorted(d.glob("*.png")):
            out.append((p.stem, spec[cls]["content"], p))

    if ids is not None:
        by_id = {i: t for t in out for i in [t[0]]}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(
                f"這些影像 id 不在 {root} 裡：{missing}。"
                f"可用的是 {sorted(by_id)}")
        return [by_id[i] for i in ids]
    return out


class MaskEditor:
    """一個 Tk 視窗，逐張畫、逐張存。

    遮罩本身是一張 PIL `L` 影像（0／255），畫布上顯示的是它與原圖的疊合。
    每一筆操作前先把當前遮罩推進 `undo` 堆疊——整張複製，因為 512² 的
    8-bit 影像只有 256 KB，記錄差分的複雜度換不到什麼。
    """

    def __init__(self, targets, out_dir: Path):
        self.targets = targets
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.brush = 24
        self.undo: List[Image.Image] = []
        # 套索：拖曳期間累積的軌跡點。放開時封閉成多邊形並填滿。
        self.lasso: List[Tuple[int, int]] = []
        # 多邊形：逐次點擊累積的頂點，Enter 或雙擊封閉。
        self.poly: List[Tuple[int, int]] = []
        self.cursor: Optional[Tuple[int, int]] = None
        self.poly_mode = False
        self.saved_any = False

        self.root = tk.Tk()
        self.root.title("draw_masks")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, highlightthickness=0, cursor="cross")
        self.canvas.pack()
        self.status = tk.Label(self.root, anchor="w", font=("Consolas", 10),
                               justify="left")
        self.status.pack(fill="x")

        c = self.canvas
        c.bind("<Button-1>", self._press)
        c.bind("<B1-Motion>", self._drag)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<Double-Button-1>", self._close_poly)
        c.bind("<Motion>", self._motion)
        c.bind("<Button-3>", self._paint)
        c.bind("<B3-Motion>", self._paint)
        self.root.bind("<Key>", self._key)
        self.root.bind("<Control-z>", lambda e: self._undo())

        self._load(0)

    # ---- 影像與遮罩 ----

    def _load(self, i: int) -> None:
        self.idx = i
        self.image_id, self.content, path = self.targets[i]
        self.src = Image.open(path).convert("RGB")
        self.w, self.h = self.src.size
        existing = self.out_dir / f"{self.image_id}.png"
        if existing.exists():
            m = Image.open(existing).convert("L")
            if m.size != self.src.size:
                raise SystemExit(
                    f"{existing} 尺寸 {m.size} 與原圖 {self.src.size} 不符。"
                    "遮罩必須與原圖同尺寸，否則對位是猜的")
            self.mask = m
        else:
            self.mask = Image.new("L", self.src.size, 0)
        self.undo.clear()
        self.canvas.config(width=self.w, height=self.h)
        self._redraw()

    def _coverage(self) -> float:
        h = self.mask.histogram()
        return sum(h[128:]) / float(self.w * self.h)

    def _redraw(self) -> None:
        # 疊合：遮罩內的像素往 OVERLAY_RGB 靠 OVERLAY_ALPHA。半透明是必要的
        # ——實心色塊會蓋掉底下的內容，而「遮罩有沒有壓到主體」正是要看的。
        tint = Image.new("RGB", self.src.size, OVERLAY_RGB)
        alpha = self.mask.point(lambda v: int(v * OVERLAY_ALPHA))
        shown = self.src.copy()
        shown.paste(tint, (0, 0), alpha)

        self.photo = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        # 描邊中的預覽。實線是已經走過的軌跡，虛線是封閉時會補上的那一段
        # ——不畫它的話，看不出鬆手會圍出什麼形狀。
        if len(self.lasso) > 1:
            self.canvas.create_line(*[c for p in self.lasso for c in p],
                                    fill="#00ff88", width=2)
            self.canvas.create_line(*self.lasso[-1], *self.lasso[0],
                                    fill="#00ff88", width=1, dash=(4, 3))
        if self.poly:
            for a, b in zip(self.poly, self.poly[1:]):
                self.canvas.create_line(*a, *b, fill="#00ff88", width=2)
            if self.cursor:
                self.canvas.create_line(*self.poly[-1], *self.cursor,
                                        fill="#00ff88", width=1, dash=(4, 3))
                self.canvas.create_line(*self.cursor, *self.poly[0],
                                        fill="#00ff88", width=1, dash=(2, 4))
            for x, y in self.poly:
                self.canvas.create_rectangle(x - 3, y - 3, x + 3, y + 3,
                                             outline="#00ff88", fill="#003322")

        cov = self._coverage()
        flag = "" if COVERAGE_WINDOW[0] <= cov <= COVERAGE_WINDOW[1] else \
            f"  ← 在窗口 {COVERAGE_WINDOW} 之外"
        mode = (f"多邊形（{len(self.poly)} 點，Enter／雙擊封閉）"
                if self.poly_mode else "套索")
        self.status.config(
            text=(f"[{self.idx + 1}/{len(self.targets)}] {self.image_id}   "
                  f"c_a={self.content!r}（要留在遮罩外）   "
                  f"涵蓋率={cov:.3f}{flag}\n"
                  f"工具={mode}（V 切換） · 左鍵沿輪廓描 · "
                  f"右鍵筆刷({self.brush}px)修邊 · Shift 擦除 · [ ] 調筆刷\n"
                  f"Ctrl+Z 復原 · C 清空 · N 存檔並下一張 · P 上一張 · "
                  f"Esc 取消描邊／離開"))

    def _push_undo(self) -> None:
        self.undo.append(self.mask.copy())
        if len(self.undo) > 40:
            self.undo.pop(0)

    def _undo(self) -> None:
        if self.undo:
            self.mask = self.undo.pop()
            self._redraw()

    # ---- 滑鼠 ----

    def _fill_path(self, pts, erase: bool) -> None:
        """把一條封閉路徑填進遮罩。少於三點不成面，直接忽略。

        除了 `polygon` 之外還沿路徑描一條 `line`：多邊形填色對只有一兩個
        像素寬的凹處會漏掉，而描邊補得回來。兩者用同一個值，故擦除也適用。
        """
        if len(pts) < 3:
            return
        self._push_undo()
        val = 0 if erase else 255
        d = ImageDraw.Draw(self.mask)
        d.polygon(pts, fill=val)
        d.line(list(pts) + [pts[0]], fill=val, width=2)

    def _press(self, e) -> None:
        if self.poly_mode:
            self.poly.append((e.x, e.y))
        else:
            self.lasso = [(e.x, e.y)]
        self._redraw()

    def _drag(self, e) -> None:
        if self.poly_mode:
            return
        # 只在移動超過一個像素時才記點：Tk 的 Motion 事件很密，全記下來會讓
        # 預覽線變成一團而看不出描到哪裡。
        if not self.lasso or (e.x, e.y) != self.lasso[-1]:
            self.lasso.append((e.x, e.y))
        self._redraw()

    def _release(self, e) -> None:
        if self.poly_mode or not self.lasso:
            return
        pts, self.lasso = self.lasso, []
        # 點一下就放開（沒有拖出軌跡）不該清掉畫面，也不該填任何東西。
        self._fill_path(pts, erase=bool(e.state & 0x0001))
        self._redraw()

    def _motion(self, e) -> None:
        """多邊形模式下要有牽引線，否則看不出下一段會連到哪裡。"""
        self.cursor = (e.x, e.y)
        if self.poly_mode and self.poly:
            self._redraw()

    def _close_poly(self, e=None) -> None:
        if not self.poly:
            return
        pts, self.poly = self.poly, []
        shift = bool(e.state & 0x0001) if e is not None else False
        self._fill_path(pts, erase=shift)
        self._redraw()

    def _paint(self, e) -> None:
        # 連續拖曳只在第一筆推 undo：整段塗抹是一個操作。
        if getattr(self, "_painting", None) is not e.state:
            self._push_undo()
            self._painting = e.state
        r = self.brush // 2
        val = 0 if (e.state & 0x0001) else 255
        ImageDraw.Draw(self.mask).ellipse(
            (e.x - r, e.y - r, e.x + r, e.y + r), fill=val)
        self._redraw()

    # ---- 鍵盤 ----

    def _key(self, e) -> None:
        k = e.keysym.lower()
        if k in ("return", "kp_enter") and self.poly:
            # 描邊中的 Enter 是「封閉這個多邊形」，不是「換下一張」。
            self._close_poly()
        elif k in ("return", "kp_enter", "n"):
            self._save()
            if self.idx + 1 < len(self.targets):
                self._load(self.idx + 1)
            else:
                print("最後一張已存檔。")
                self.root.destroy()
        elif k == "v":
            self.poly_mode = not self.poly_mode
            self.poly, self.lasso = [], []
            self._redraw()
        elif k == "p":
            if self.idx > 0:
                self._save()
                self._load(self.idx - 1)
        elif k == "c":
            self._push_undo()
            self.mask = Image.new("L", self.src.size, 0)
            self._redraw()
        elif k == "bracketleft":
            self.brush = max(4, self.brush - 6)
            self._redraw()
        elif k == "bracketright":
            self.brush = min(200, self.brush + 6)
            self._redraw()
        elif k == "escape":
            # 描邊途中的 Esc 是「取消這一筆」。直接關掉視窗會讓一個手滑
            # 點出來的多邊形變成「不小心離開」，而畫了一半的圖沒存。
            if self.poly or self.lasso:
                self.poly, self.lasso = [], []
                self._redraw()
            else:
                self.root.destroy()

    def _save(self) -> None:
        cov = self._coverage()
        if cov == 0.0:
            print(f"  [skip] {self.image_id} 遮罩為空，不存檔")
            return
        p = self.out_dir / f"{self.image_id}.png"
        self.mask.save(p)
        self.saved_any = True
        flag = "" if COVERAGE_WINDOW[0] <= cov <= COVERAGE_WINDOW[1] else \
            "  ← 在可用窗口之外，由段 0 判讀決定收不收"
        print(f"  [save] {p}  涵蓋率={cov:.4f}{flag}")

    def run(self) -> None:
        self.root.mainloop()


def invert_masks(targets, out_dir: Path, guard: int = 13) -> List[tuple]:
    """把「描在主體上」的遮罩翻成「主體之外」的遮罩。

    描主體比描背景快得多也準得多——輪廓是看得見的，背景的邊界不是。故作法
    是先描主體再翻面，而不是要求直接描背景。

    翻面不是單純取補集。補集會**緊貼**主體輪廓，而式 (4) 的 M 定義在 64²
    的注意力格點上（512² 影像下一格等於 8×8 像素），輪廓上任何一格都同時
    含有主體與補集的像素，`assert_masks_disjoint` 以 max-pool 比對時必然
    判為重疊。故先把主體膨脹 `guard` 像素再取補集，兩者之間留一條帶。

    **這條帶只是降低風險，不構成保證**：M 是 c_a 的**注意力**區域，可能比
    肉眼看到的物件更大或偏移，而本機沒有模型算不出它。具約束力的檢查仍在
    `src/defense/optimize.py`，跑段 0 時才會執行。

    原稿存到 `<out_dir>/_subject/`，且**之後一律以原稿為輸入**——重跑本
    函式不會把已經翻過的再翻一次。
    """
    from PIL import ImageFilter

    sub_dir = out_dir / "_subject"
    sub_dir.mkdir(parents=True, exist_ok=True)
    k = guard if guard % 2 else guard + 1

    rows = []
    for image_id, content, _ in targets:
        cur = out_dir / f"{image_id}.png"
        keep = sub_dir / f"{image_id}.png"
        if keep.exists():
            m = Image.open(keep).convert("L")       # 冪等：永遠從原稿翻
        elif cur.exists():
            m = Image.open(cur).convert("L")
            m.save(keep)
        else:
            rows.append((image_id, content, None, None))
            continue

        before = sum(m.histogram()[128:]) / float(m.size[0] * m.size[1])
        grown = m.filter(ImageFilter.MaxFilter(k)) if k >= 3 else m
        inv = grown.point(lambda v: 0 if v >= 128 else 255)
        inv.save(cur)
        after = sum(inv.histogram()[128:]) / float(inv.size[0] * inv.size[1])
        rows.append((image_id, content, before, after))
    return rows


def _coverage_of(mask_img: "Image.Image") -> float:
    w, h = mask_img.size
    return sum(mask_img.histogram()[128:]) / float(w * h)


def _invert_one(subject: "Image.Image", guard: int) -> "Image.Image":
    """單張主體遮罩 → 翻面後的攻擊遮罩。與 `invert_masks` 用同一條路徑。"""
    from PIL import ImageFilter

    k = guard if guard % 2 else guard + 1
    grown = subject.filter(ImageFilter.MaxFilter(k)) if k >= 3 else subject
    return grown.point(lambda v: 0 if v >= 128 else 255)


def recrop(targets, orig_root: Path, out_dir: Path, *, target: float = 0.30,
           guard: int = 13, size: int = 512, max_upscale: float = 3.0,
           skip_in_window: bool = True) -> List[dict]:
    """裁切放大原圖，使翻面後的遮罩涵蓋率落進可用窗口。

    翻面後 `涵蓋率 = 1 − 主體佔比 − 保護帶`，故主體愈小、遮罩愈大。要把
    涵蓋率壓進 [0.15, 0.45]，等價於讓主體在畫面中佔到約 0.55–0.85——對
    主體很小的照片，唯一的作法是裁掉多餘的背景再放大。

    **裁切邊長以二分搜尋決定，不用解析式。** 保護帶是對主體做形態學膨脹，
    它佔的比例隨主體的周長而變、不是面積的函數，解析式會錯；直接對「裁完
    翻完之後的涵蓋率」搜尋則自動把它算進去。

    約束：

    - 裁切框必須**完整含住主體**（下界取主體外接框的長邊），否則等於把要
      保住的物件切掉一塊。
    - 裁切框為正方形並貼齊影像邊界，中心取主體外接框的中心後夾回範圍內。
    - 放大倍率上限 `max_upscale`。超過就取最小可用的框並回報實際涵蓋率——
      **不為了湊數字把圖放大到糊掉**，那會改變保真基線而與防禦無關。

    原圖備份到 `<orig_root>/<類別>/`，且之後一律以備份為輸入，
    故重跑不會愈裁愈小。回傳逐影像的紀錄供落盤。
    """
    sub_dir = out_dir / "_subject"
    rows: List[dict] = []

    for image_id, content, path in targets:
        sp = sub_dir / f"{image_id}.png"
        if not sp.exists():
            rows.append({"image_id": image_id, "status": "沒有主體遮罩"})
            continue

        orig_dir = orig_root / path.parent.name
        orig_dir.mkdir(parents=True, exist_ok=True)
        backup = orig_dir / path.name
        if not backup.exists():
            Image.open(path).convert("RGB").save(backup)
        src = Image.open(backup).convert("RGB")

        # 主體遮罩也要從未裁切的版本來。第一次裁切前先留底。
        sub_backup = sub_dir / "_full" / f"{image_id}.png"
        sub_backup.parent.mkdir(parents=True, exist_ok=True)
        if not sub_backup.exists():
            Image.open(sp).convert("L").save(sub_backup)
        sub = Image.open(sub_backup).convert("L")
        if sub.size != src.size:
            sub = sub.resize(src.size, Image.NEAREST)

        cov_now = _coverage_of(_invert_one(sub, guard))
        if skip_in_window and COVERAGE_WINDOW[0] <= cov_now <= COVERAGE_WINDOW[1]:
            rows.append({"image_id": image_id, "status": "已在窗口內",
                         "coverage": round(cov_now, 4)})
            continue

        W, H = src.size
        bbox = sub.point(lambda v: 255 if v >= 128 else 0).getbbox()
        if bbox is None:
            rows.append({"image_id": image_id, "status": "主體遮罩為空"})
            continue
        x0, y0, x1, y1 = bbox
        need = max(x1 - x0, y1 - y0)                 # 必須含住主體
        floor = max(need, int(round(size / max_upscale)))
        lo, hi = min(floor, min(W, H)), min(W, H)

        def crop_at(side: int):
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            half = side // 2
            px = min(max(cx - half, 0), W - side)
            py = min(max(cy - half, 0), H - side)
            box = (px, py, px + side, py + side)
            s = sub.crop(box).resize((size, size), Image.NEAREST)
            return box, s

        # 邊長愈小 → 主體佔比愈大 → 翻面後涵蓋率愈小，對邊長單調遞增。
        best = None
        a, b = lo, hi
        for _ in range(14):
            mid = (a + b) // 2
            if mid < lo:
                break
            box, s = crop_at(mid)
            cov = _coverage_of(_invert_one(s, guard))
            if best is None or abs(cov - target) < abs(best[2] - target):
                best = (mid, box, cov)
            if cov > target:
                b = mid - 1
            else:
                a = mid + 1
            if a > b:
                break

        side, box, cov = best
        scale = size / side
        img = src.crop(box).resize((size, size), Image.LANCZOS)
        img.save(path)
        sub_cropped = crop_at(side)[1]
        sub_cropped.save(sp)
        # 攻擊遮罩也要一起重寫。只更新原圖與主體遮罩的話，`masks/*.png` 會
        # 留著裁切前那一張，疊在新圖上錯位——而它正是實驗真正讀進去的檔案。
        _invert_one(sub_cropped, guard).save(out_dir / f"{image_id}.png")

        # 沒到目標時，是哪一個下界擋住的？兩者的處置完全不同：放大上限可以
        # 放寬，外接框不行——那是主體形狀的性質。把它們混為一談會讓人去調
        # 一個調了也沒用的旋鈕。
        if cov <= COVERAGE_WINDOW[1]:
            limit = ""
        elif need >= int(round(size / max_upscale)):
            # 外接框內主體只佔這麼多，裁到貼齊外接框也不會更高
            solidity = float((sub.point(lambda v: 255 if v >= 128 else 0)
                              .histogram()[255])) / max(need * need, 1)
            limit = f"外接框（框內主體僅佔 {solidity:.2f}）"
        else:
            limit = "放大上限"
        rows.append({"image_id": image_id, "status": "已裁切",
                     "content": content, "crop_box": str(box),
                     "crop_side": side, "upscale": round(scale, 2),
                     "coverage_before": round(cov_now, 4),
                     "coverage": round(cov, 4), "limited_by": limit})
    return rows


def render_overview(targets, out_dir: Path, thumb: int = 256, cols: int = 4
                    ) -> Path:
    """把全部遮罩疊在各自的原圖上排成一張總覽圖，供人眼一次看完。

    涵蓋率是數字，「遮罩有沒有貼合輪廓」「c_a 有沒有被壓到」不是——後兩者
    只能看。判準以人眼為主、數值為輔（`CLAUDE.md`），故這張圖是選圖的
    主要依據，涵蓋率只是標在旁邊的參考。

    缺遮罩的影像照樣列出並標明，否則「還沒畫」與「畫了但很小」在圖上分不出來。
    """
    from PIL import ImageDraw as _ID

    pad, bar = 8, 22
    rows = (len(targets) + cols - 1) // cols
    W = cols * (thumb + pad) + pad
    H = rows * (thumb + bar + pad) + pad
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    draw = _ID.Draw(sheet)

    for i, (image_id, content, path) in enumerate(targets):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + bar + pad)

        src = Image.open(path).convert("RGB")
        mp = out_dir / f"{image_id}.png"
        if mp.exists():
            m = Image.open(mp).convert("L").resize(src.size, Image.NEAREST)
            cov = sum(h for h in m.histogram()[128:]) / float(
                src.size[0] * src.size[1])
            tint = Image.new("RGB", src.size, OVERLAY_RGB)
            shown = src.copy()
            shown.paste(tint, (0, 0), m.point(
                lambda v: int(v * OVERLAY_ALPHA)))
            inw = COVERAGE_WINDOW[0] <= cov <= COVERAGE_WINDOW[1]
            label = f"{image_id}  c_a={content}  {cov:.3f}"
            colour = (120, 255, 160) if inw else (255, 190, 90)
        else:
            shown = src.copy()
            label = f"{image_id}  c_a={content}  （未繪製）"
            colour = (255, 90, 90)

        sheet.paste(shown.resize((thumb, thumb), Image.LANCZOS), (x, y))
        draw.text((x + 2, y + thumb + 5), label, fill=colour)

    p = out_dir / "overview.png"
    sheet.save(p)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--images", nargs="+", default=None,
                    help="只畫這幾張；不給時全部 24 張")
    ap.add_argument("--out", type=Path, default=Path("data/lo_masks"),
                    help="遮罩輸出目錄。**不可放在資料集目錄裡**："
                         "`load_lo_aligned` 拒絕未宣告卻有 PNG 的子目錄，"
                         "那道檢查擋的是「忘了宣告類別」，不該為遮罩鬆綁")
    ap.add_argument("--overview", action="store_true",
                    help="不開視窗，只把現有遮罩排成一張總覽圖供人眼判讀")
    ap.add_argument("--invert", action="store_true",
                    help="把描在主體上的遮罩翻成主體之外的遮罩（先膨脹出"
                         "保護帶再取補集）。原稿留在 <out>/_subject/，"
                         "重跑一律從原稿翻，不會翻兩次")
    ap.add_argument("--originals", type=Path, default=Path("data/lo_original"),
                    help="裁切前原圖的備份目錄，同樣不可放在資料集目錄裡")
    ap.add_argument("--recrop", action="store_true",
                    help="裁切放大原圖，使翻面後的涵蓋率落進可用窗口。"
                         "原圖備份在 <data>/_original/，一律從備份裁，"
                         "重跑不會愈裁愈小。已在窗口內的影像不動")
    ap.add_argument("--target", type=float, default=0.30,
                    help="裁切要命中的遮罩涵蓋率，預設取窗口中點")
    ap.add_argument("--max-upscale", type=float, default=3.0,
                    help="放大倍率上限。超過就取最小可用的框並回報實際值——"
                         "不為了湊數字把圖放大到糊掉")
    ap.add_argument("--size", type=int, default=512,
                    help="裁切後輸出的邊長，須與批次的 --resolution 一致")
    ap.add_argument("--guard", type=int, default=13,
                    help="翻面時主體要膨脹幾個像素。512 影像下注意力格點"
                         "一格等於 8 像素，預設 13 約留一格半")
    args = ap.parse_args()

    targets = load_targets(args.data, args.images)
    out_dir = args.out

    if args.recrop:
        rows = recrop(targets, args.originals, out_dir, target=args.target,
                      guard=args.guard, size=args.size,
                      max_upscale=args.max_upscale)
        print(f"裁切目標涵蓋率 {args.target}，放大上限 {args.max_upscale}×，"
              f"原圖備份在 {args.originals}")
        for r in rows:
            if r["status"] != "已裁切":
                print(f"  [{r['status']}] {r['image_id']}"
                      + (f"  涵蓋率={r['coverage']:.3f}" if "coverage" in r
                         else ""))
                continue
            flag = ("" if not r.get("limited_by")
                    else f"  ← 仍在窗口外，卡在{r['limited_by']}")
            print(f"  [crop] {r['image_id']:10s} {r['crop_side']:>3d}px "
                  f"放大 {r['upscale']:.2f}×  "
                  f"涵蓋率 {r['coverage_before']:.3f} → "
                  f"{r['coverage']:.3f}{flag}")
        import csv as _csv

        cp = out_dir / "recrop.csv"
        keys = ["image_id", "status", "content", "crop_box", "crop_side",
                "upscale", "coverage_before", "coverage", "limited_by"]
        with cp.open("w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"紀錄：{cp}")
        render_overview(targets, out_dir)
        print(f"總覽圖：{out_dir / 'overview.png'}")
        return

    if args.invert:
        rows = invert_masks(targets, out_dir, guard=args.guard)
        print(f"翻面完成，保護帶 {args.guard} px，原稿在 {out_dir / '_subject'}")
        for image_id, content, before, after in rows:
            if before is None:
                print(f"  [skip] {image_id} 沒有遮罩")
                continue
            flag = "" if COVERAGE_WINDOW[0] <= after <= COVERAGE_WINDOW[1] \
                else "  ← 在可用窗口之外"
            print(f"  [invert] {image_id:10s} c_a={content:6s} "
                  f"主體 {before:.3f} → 遮罩 {after:.3f}{flag}")
        render_overview(targets, out_dir)
        print(f"總覽圖：{out_dir / 'overview.png'}")
        return

    if args.overview:
        p = render_overview(targets, out_dir)
        print(f"總覽圖：{p}")
        return

    print(f"{len(targets)} 張待畫，輸出到 {out_dir}")
    print("c_a 要留在遮罩外——那是 Lo Figure 3 的配置，也是 DEF-011 的處置。")
    MaskEditor(targets, out_dir).run()


if __name__ == "__main__":
    main()
