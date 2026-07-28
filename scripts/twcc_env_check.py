"""TWCC 容器環境健檢 — 確認 conda env 的 torch 能在 V100 上實際執行 CUDA op。

容器預裝的 NGC torch 沒有 build sm_70，且 ldconfig 指向不相容的 forward-compat
libcuda，兩者都會讓 CUDA op 失敗但 `torch.cuda.is_available()` 仍為 True。
因此健檢必須實際跑一次 matmul 與一次 conv，不能只看 is_available()。

執行：source env.sh && python scripts/twcc_env_check.py
"""

import sys

import torch


def main():
    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}")
    print(f"cuda avail  {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: 無 CUDA 裝置")
        return 1

    p = torch.cuda.get_device_properties(0)
    print(f"gpu         {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 1024**3:.1f} GB")
    print(f"arch_list   {torch.cuda.get_arch_list()}")

    a = torch.randn(512, 512, device="cuda")
    print(f"matmul      ok  {float((a @ a).sum()):.3f}")

    # conv 走 cuDNN，與 matmul 走的是不同的函式庫，需分別驗證
    x = torch.randn(1, 4, 64, 64, device="cuda")
    w = torch.randn(4, 4, 3, 3, device="cuda")
    y = torch.nn.functional.conv2d(x, w, padding=1)
    print(f"cudnn conv  ok  {tuple(y.shape)}")

    for name in ("diffusers", "transformers", "torchvision"):
        try:
            mod = __import__(name)
            print(f"{name:11s} {mod.__version__}")
        except ImportError:
            print(f"{name:11s} MISSING")
    return 0


if __name__ == "__main__":
    sys.exit(main())
