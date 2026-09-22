# 環境與操作

卡的規則見 `CLAUDE.md` 的 GPU 一節。

## 遠端

GPU 工作一律在 NYCU BASIC lab 跑，兩台各 8 張 RTX 3090，home 目錄跨機同步。

```
ssh -p 10101 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-1
ssh -p 10102 nelson0314@server.basiclab.lab.nycu.edu.tw   # basic-2
```

金鑰認證已設好。

Repo 在 `/nfs/home/nelson0314/image-immunization`（由本機整棵同步過去，不是 git 工作區）。
虛擬環境由 `~/env.sh` 提供（`$PY` 指向 `~/venvs/wacv/bin/python`，另設
`HF_HOME`、`PYTHONPATH`）。`PYTHONPATH` 指向舊的 `~/WACV`，派工腳本則自己把
repo 根目錄插在 `sys.path[0]`，兩棵樹同時在路徑上。
套件安裝走 `uv`：`VIRTUAL_ENV=$VENV uv pip install <pkg>`（venv 裡沒有 pip）。

**先 `source ~/env.sh`，再 `cd` 到 repo。** `env.sh` 最後一行會把工作目錄換到
舊的 `~/WACV`，順序相反時所有相對路徑指向錯的樹，而且不會報錯。

HF 權重走 `~/env.sh` 設的 `HF_HOME=$HOME/hf_cache`（NFS 上，跨機共用）。
`/var/cache/huggingface/hub` 是機器本地的另一份快取，不是本專案在用的那份。
