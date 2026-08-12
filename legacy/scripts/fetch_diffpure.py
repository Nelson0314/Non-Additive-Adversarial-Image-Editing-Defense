"""取得 DiffPure 所需的檢查點與套件。

    python scripts/fetch_diffpure.py --dest /work/hf_cache/diffpure

DiffPure（`diffpure` 淨化算子）需要兩件本 repo **不入版控**的東西：

1. `256x256_diffusion_uncond.pt` —— guided-diffusion 的 256×256 無條件模型，
   2.2 GB。它是第三方權重不是本專案的實驗證據，故不進 `runs/`。
2. `guided_diffusion` 套件 —— OpenAI 的 repo。

## 為什麼套件要用 editable 安裝

上游 `setup.py` 寫的是 `py_modules=["guided_diffusion"]`。`py_modules` 指的是
**單一 .py 檔**，而 `guided_diffusion` 是一個目錄，故

    pip install git+https://github.com/openai/guided-diffusion.git

會「安裝成功」但只裝進 metadata（實測 wheel 僅 1978 bytes），`import
guided_diffusion` 仍然失敗。這是上游的缺陷，不是環境問題。可用的作法是
clone 之後 `pip install -e`：editable 安裝把來源目錄放進 `sys.path`，
真正的套件目錄因此可見。本腳本照此執行。

## 之後怎麼用

把檢查點路徑放進環境變數：

    export DIFFPURE_CKPT=<dest>/256x256_diffusion_uncond.pt

`src/purify/diffpure.py::has_diffpure_weights` 會同時檢查該檔與套件是否到位；
兩者缺一時 `Purifier("diffpure").available` 為 False，`annotate_unavailable`
會在**跑之前**把相關格點標成 skipped，而不是跑到那一格才炸。
"""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.purify.diffpure import (  # noqa: E402
    DIFFPURE_CHECKPOINT, DIFFPURE_CKPT_ENV,
)

CKPT_URL = ("https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
            "256x256_diffusion_uncond.pt")
CKPT_BYTES = 2211383297          # 2026-08-05 由 HTTP Content-Length 核對
REPO_URL = "https://github.com/openai/guided-diffusion.git"


def download(url: str, dest: Path, expect_bytes: int) -> Path:
    if dest.exists() and dest.stat().st_size == expect_bytes:
        print(f"[skip] 已存在且大小相符：{dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}\n       → {dest}（{expect_bytes / 1e9:.2f} GB）")
    got = 0
    with urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if got % (1 << 26) < (1 << 20):
                print(f"       {got / 1e9:.2f} / {expect_bytes / 1e9:.2f} GB",
                      flush=True)
    if got != expect_bytes:
        # 半份權重載入時會以形狀不符中止，訊息指不到根本原因。此處先擋下。
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"下載長度 {got} 與預期 {expect_bytes} 不符")
    tmp.replace(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_package(src_dir: Path) -> None:
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (src_dir / "guided_diffusion").is_dir():
        print(f"[git ] clone {REPO_URL} → {src_dir}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL,
                        str(src_dir)], check=True)
    print(f"[pip ] install -e {src_dir}（上游 setup.py 的 py_modules 缺陷，見 docstring）")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e",
                    str(src_dir)], check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, required=True,
                    help="檢查點與套件的存放目錄（不要放在 repo 內）")
    ap.add_argument("--skip-package", action="store_true")
    ap.add_argument("--sha256", action="store_true",
                    help="下載後計算雜湊（2.2 GB，約需一分鐘）")
    args = ap.parse_args(argv)

    ckpt = download(CKPT_URL, args.dest / DIFFPURE_CHECKPOINT, CKPT_BYTES)
    if args.sha256:
        print(f"[hash] sha256 {sha256(ckpt)}")
    if not args.skip_package:
        install_package(args.dest / "guided-diffusion")

    print("\n完成。請設定環境變數後再跑段 3：")
    print(f"  export {DIFFPURE_CKPT_ENV}={ckpt}")
    if os.environ.get(DIFFPURE_CKPT_ENV) != str(ckpt):
        print(f"  （目前 {DIFFPURE_CKPT_ENV}="
              f"{os.environ.get(DIFFPURE_CKPT_ENV, '未設定')}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
