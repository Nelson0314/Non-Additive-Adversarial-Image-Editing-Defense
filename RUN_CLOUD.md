# RUN_CLOUD.md — Lightning AI（H100 80GB）方向可行性執行指南

本指南用於在 Lightning AI Studio 的 H100 80GB 上，以**真實 Stable Diffusion**
跑通 stage0 → stage1 → stage2 全流程，確認「非加性 vs 加性保護在淨化後之相對
行為」方向是否可行。**此為可行性驗證，非正式實驗**：規模刻意縮小、預設使用
placeholder 合成資料集（`configs/base.yaml` 之 `is_placeholder: true`），
結果不與 DAYN Table 1 或任何外部基準比較。正式實驗規模見 `PREFLIGHT.md` [5]。

---

## 0. 前置

- 一個 Lightning AI Studio，GPU 選 **H100 80GB**。
- 一個 Hugging Face 帳號（下載 SD 權重用；下方步驟 3 說明授權）。
- 本專案已推上 GitHub（若尚未，見本檔最後「附錄 A：推上 GitHub」）。

---

## 1. 取得程式碼

在 Studio 的終端機：

```bash
git clone https://github.com/<你的帳號>/<你的 repo>.git WACV
cd WACV
```

> `external/`、`experiments/`、`data/dayn_testset/` 依 `.gitignore` **不在** repo 內。
> 核心流程用 placeholder 資料即可跑，不需 DAYN 資料集。真實 GrIDPure 需另行
> 下載（步驟 6，可選）。

---

## 2. 建立 Python 環境

Lightning Studio 通常已預裝 CUDA 版 `torch`。**不要重裝 torch**（會覆蓋成錯誤
build），只裝其餘依賴：

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# 應印出 cuda True；若 False 表示這台 Studio 沒掛到 GPU

pip install diffusers transformers accelerate piq opencv-contrib-python peft pyyaml matplotlib pytest psutil
```

> **numpy 2 相依坑（Lightning cloudspace 環境必做）**：上面的安裝會把 numpy 升到
> 2.x，但 Lightning 基礎環境預裝的 `scipy`/`scikit-learn`/`pandas` 是針對 numpy 1.x
> 編譯的，於是 `import diffusers` 會沿 `transformers → scipy/sklearn` 鏈報
> `cannot import name 'Inf' from 'numpy'` 或 `numpy.dtype size changed`。把這三個
> 套件升級到 numpy-2 相容版本即可（保留 numpy 2，勿降版——opencv 5 需要 numpy 2）：
>
> ```bash
> pip install -U "scipy>=1.13" scikit-learn pandas
> python -c "from diffusers.pipelines.stable_diffusion import pipeline_stable_diffusion; print('IMPORT OK')"
> ```

驗證裝置與單元測試（首次會下載 tiny 測試模型，約數十 MB）：

```bash
python -m src.utils.device        # 應顯示 backend: cuda
python -m pytest tests/ -q        # 預期 44 passed, 1 skipped
```

`test_edit_batching.py` 的位元級啟用閘門在 `edit_batch_size=1` 時會 skip（正常）。
批次化容差測試（`test_*_within_tolerance`）在 GPU 上以 1e-2 判定（kernel/TF32 隨
batch 之數值差較 CPU 大，屬正常）。

---

## 3. 取得 SD 權重授權（Hugging Face）

`CompVis/stable-diffusion-v1-4` 與 `stabilityai/stable-diffusion-2-base` 為
gated 模型，須先於 HF 網頁接受授權，再於 Studio 登入：

1. 瀏覽器登入 huggingface.co，到下列頁面各按一次 **Agree/Access**：
   - https://huggingface.co/CompVis/stable-diffusion-v1-4
   - https://huggingface.co/stabilityai/stable-diffusion-2-base
2. 在 Studio 終端機以 token 登入（HF 設定頁 → Access Tokens 建一個 read token）：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login    # 貼上 token
```

> 若不想處理 gated 授權，可行性驗證亦可改用非 gated 模型跑通流程（方向性即可），
> 例如把下方指令的 `CompVis/stable-diffusion-v1-4` 換成
> `stabilityai/stable-diffusion-2-1-base`（同樣 gated）或任何本機已可存取的 SD1.x
> 權重。模型名稱只透過 `--model` / `--protect-model` 傳入，程式不硬編碼。

---

## 4. 可行性規模（兩檔速度）

流程支援命令列覆寫，不需改 YAML。兩個關鍵旋鈕決定時間：

- **影像數**：`--max-images N`（placeholder 資料每張含 2 個 prompt）。
- **保護方法**：`--methods`。`pg_diff`（PhotoGuard diffusion attack，EOT×10）
  是**單一最大成本**（每張約 15 分鐘 @H100，估計值），其餘方法每張 <1 分鐘。

| 檔位 | 方法集 | 影像/seed | 預估時間（H100，估計值） | 能確認 |
|---|---|---|---|---|
| **快速**（建議先跑） | `pg_enc,advdiff,apa,hybrid`（略 pg_diff） | 6 張 / 3 seed | **~40–60 分鐘** | 流程跑通、非加性 vs encoder-attack 之 drop 方向 |
| **完整方法** | 全 5 法（含 pg_diff） | 6 張 / 3 seed | **~3–4 小時**（pg_diff 保護 ~1.5h 主宰） | 加上 diffusion-attack 基準 |

> 時間為估計值。第一次執行時 stage1 會逐張印出各方法保護耗時（`protect ... Ns`），
> 以此校正你對總時數的預期。

（可選加速）H100 記憶體充裕，編輯可批次化以縮短 stage1/stage2 的編輯時間。
可行性驗證不要求逐位元重現，故可放心啟用：

```bash
# 一次性把編輯批次大小設為 8（噪聲協定與 batch=1 一致，僅 kernel 數值差 ~1e-5，
# 對可行性判讀無影響；細節見 NOTES.md 2026-07-24 批次化條目）
sed -i 's/^\(  edit_batch_size:\) 1/\1 8/' configs/base.yaml
```

---

## 5. 執行流程（快速檔為例）

以下用單一模型（v1-4）、單一編輯（sdedit）、`strength=0.8`（diffusers 預設值，
SPEC §2.8 v5 首選）。輸出落在 `experiments/<stage>/<timestamp>/`。

```bash
export MODEL=CompVis/stable-diffusion-v1-4

# (可選) stage0：相似性校準，寫出各非加性方法之 eps 到 configs/nonadditive_calibrated.yaml
#         略 pg_diff 基準以省時（--skip-pg-diff）；stage1 會自動合併校準值
python scripts/stage0_calibrate.py \
    --model $MODEL --max-images 6 --skip-pg-diff

# stage1：乾淨情況比較（保護 → 同 seed 編輯原圖與保護圖 → 指標）
python scripts/stage1_clean.py \
    --protect-model $MODEL --model $MODEL \
    --methods pg_enc,advdiff,apa,hybrid \
    --edit-methods sdedit \
    --max-images 6 --n-seeds 3 \
    --sdedit-strength 0.8 --with-clip
# 完成後記下印出的 run 目錄，例如 experiments/stage1/20260724_hhmmss

# stage2：淨化後比較（讀 stage1 之受保護影像；輕量淨化，先不跑真實 GrIDPure）
python scripts/stage2_purify.py \
    --stage1-dir experiments/stage1/<上一步的時間戳> \
    --purify-methods jpeg,blur,crop_resize,advclean_bf,advclean_bfgf
```

完整方法檔：把 stage1 的 `--methods` 換成
`pg_enc,pg_diff,advdiff,apa,hybrid`（其餘不變）。

---

## 6.（可選）真實 GrIDPure

輕量淨化（步驟 5）已足以看出方向。若要納入最強的 pixel-space 淨化器 GrIDPure：

```bash
# 6a. clone 官方 repo 到 external/（.gitignore 已排除，不會污染你的 repo）
mkdir -p external && git clone https://github.com/ZhengyueZhao/GrIDPure.git external/GrIDPure

# 6b. 下載無條件 guided diffusion checkpoint（約 2GB，ImageNet 256×256）
mkdir -p checkpoints
wget -O checkpoints/256x256_diffusion_uncond.pt \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

# 6c. stage2 帶入真實淨化器（只跑 gridpure 家族，接續同一個 stage1 目錄）
python scripts/stage2_purify.py \
    --stage1-dir experiments/stage1/<時間戳> \
    --purify-methods gridpure \
    --gridpure-model checkpoints/256x256_diffusion_uncond.pt
```

> 成本：GrIDPure 單張 512×512 於 V100 約 2 分鐘、H100 估計 ~0.5 分鐘。
> 6 張 × 4–5 方法 × 1 設定約 15–25 分鐘。先驗流程可改 `--gridpure-fake`
> （以 Gaussian blur 假冒淨化器，秒級完成，僅測 grid 機制與資料流）。

---

## 7. 看結果

每個 run 目錄下：

- `summary.md`：人可讀摘要。
  - stage1 依**有無 DAYN 錨點**分兩表：`sdedit`（img2img，主要條件）、
    `inpaint`（補充條件）。校準比對僅限 sdedit 之 `pg_*` 列。
  - stage2 列出各淨化法之「加性 vs 非加性 mean drop_lpips」——**方向可行性的
    核心數字**：若非加性之 drop 明顯小於加性，即支持研究假設方向。
- `results.csv`：逐列原始數據（含 `drop_*` 欄，`drop_valid` 標記）。
- `purify_strength_curve.png`（stage2）：淨化強度 vs 防禦效果曲線。
- `config_snapshot.yaml`、`env.json`：完整設定與環境（可重現）。

> 提醒：placeholder 為合成影像，數值不代表真實資料集之結論；本流程確認的是
> **方向與程式正確性**，非定量結果。要正式數字須取得 DAYN 資料集（見
> `TWCC_CHECKLIST.md`）或依 §4.3 自生成 150 張並於論文揭露。

---

## 8. 自動化執行 + 結果自動回推（機器會自動關機時用）

`scripts/run_experiment.sh` 把 stage1→stage2 串成一鍵執行：全程 stdout 存到
`lab/<時間戳>_<label>/run.log`，並在**結束時（含崩潰、被關機）自動 commit 並
push** 關鍵產出到 GitHub，機器關機後可在 `lab/` 回頭查看（見 `lab/README.md`）。

**一次性：讓雲端可 headless push**（無 TTY 不能輸入密碼，故把 PAT 併入 remote URL）：

```bash
git remote set-url origin https://<你的PAT>@github.com/Nelson0314/Non-Additive-Adversarial-Image-Editing-Defense.git
```

> PAT 會存在雲端 `.git/config`；Lightning Studio 為個人可拋式環境，可接受。
> 之後所有 `git push` 免密碼。

**執行（單行，跑完自動把結果推上來）**：

```bash
git pull && bash scripts/run_experiment.sh quick
```

可用環境變數覆寫（單行示例）：

```bash
MAXIMG=3 METHODS=pg_enc,advdiff,apa,hybrid bash scripts/run_experiment.sh quick3
```

```bash
RUN_STAGE0=1 MAXIMG=6 METHODS=pg_enc,pg_diff,advdiff,apa,hybrid bash scripts/run_experiment.sh full
```

跑完（或關機後）在本機 `git pull`，看 `lab/<時間戳>_quick/run.log` 末行
`exit=0` 與 `lab/.../stage2__*/summary.md`。

---

## 附錄 A：把本專案推上 GitHub

本機 repo 尚未設定 remote。在 github.com 建一個**空的** repo（不要勾 README/
.gitignore），取得其 URL 後，於本機（Windows PowerShell，專案根目錄）：

```powershell
git remote add origin https://github.com/<你的帳號>/<你的 repo>.git
git push -u origin main
```

> 若推送要求認證：用 GitHub 帳號 + Personal Access Token（Settings → Developer
> settings → Tokens，勾 `repo` 權限）當密碼；或先裝 GitHub CLI `gh auth login`。
> 之後每次更新：`git add -A; git commit -m "..."; git push`。

`external/`、`experiments/`、`data/dayn_testset/`、`*.ckpt`、`*.safetensors`
已在 `.gitignore`，不會被推上去——雲端依步驟 6 另行下載即可。
