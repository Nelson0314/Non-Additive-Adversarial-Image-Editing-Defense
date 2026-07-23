# 非加性 vs 加性影像保護（editing 免疫）

實作並比較「非加性」與「加性」影像保護方法，用於防止 Stable Diffusion
對影像進行惡意文字編輯。演算法規格見 `SPEC.md`，結構與介面見 `STRUCTURE.md`，
執行紀錄與決策理由見 `NOTES.md`，TWCC 部署清單見 `TWCC_CHECKLIST.md`。

## 開發與執行環境

| | 本地開發 | TWCC 執行 |
|---|---|---|
| 硬體 | CPU（Windows 11 Home，16 GB RAM） | Tesla V100-SXM2-32GB |
| 模型 | `hf-internal-testing/tiny-stable-diffusion-pipe` | SD V1.4 / V2.0 |
| 用途 | 開發與正確性驗證 | 正式實驗 |

同一份程式碼兩地通用：裝置經 `src/utils/device.py` 抽象（全專案禁用
`.cuda()`），模型名稱由 `configs/*.yaml` 提供，不硬編碼。不依賴 xformers。

## 本地環境建置

conda env `wacv`（Python 3.11；版本選擇理由與各套件實測版本見 NOTES.md）：

```powershell
conda env create -f environment.yml
conda activate wacv
```

驗證環境：

```powershell
python -m src.utils.device      # 裝置偵測，輸出附加至 NOTES.md
python -m pytest tests/ -v      # 單元測試（首次執行會下載 tiny SD 測試模型）
```

## 目錄結構

見 `STRUCTURE.md` §1。`external/`（官方 repo clone）、`experiments/`（實驗輸出）、
`data/dayn_testset/`（向作者索取之資料集）均不進版控。
