# 動物的防禦圖批次

`data/animals` 的十二張影像（`cat_00..03`、`dog_00..03`、`bird_00..03`，512²）在
各篇原生設定上求解出的受保護影像。**只有求解，沒有編輯評測。**

設定、prompt 的處置、CSV 欄位與 `runs/defence_portraits/README.md` **完全相同**，
不在這裡重複；本檔只寫與人像那一批不同的地方。

## 資料集是這一批建的

跑攻擊之前先由選單組出資料集：

    python scripts/build_dataset.py --screen runs/candidate_pick \
        --select runs/candidate_pick/selection.txt --out data/animals

選單 `runs/candidate_pick/selection.txt` 是使用者挑的 cat／dog 各四張，
加上助理建議、使用者未指定的 bird 四張。候選裁切圖在
`runs/candidate_pick/crops/`，量測欄在 `runs/candidate_pick/screen.csv`。
`data/animals/provenance.json` 逐張記下來源檔、來源 sha256 與當時量到的每一欄；
每類的 `attribution.json` 留下授權與作者。

`runs/candidate_pick/` 與 `data/_pool/{bird,cat,dog}/` 原本不在遠端，是這一批
一併同步過去的。

## 跑了哪四個條件，為什麼不是六個

只跑 `mist`、`dia_r`、`dct_shield`、`dct_shield_y`，加上佇列後段的 `dia_pt`。

**沒有對動物跑 PhotoGuard。** 兩個臂 × 十二張影像 ≈ 21 小時，而人像那一批的
PhotoGuard 還在跑。這是排程上的取捨，不是對該條件的判定。

## 沒有畫遮罩

`scripts/make_masks.py` 沒有跑。它的遮罩只被 inpainting 評測消費，而這一輪
不跑評測；且它經由 `scripts/baseline_run.py::load_dataset` 取每類的 `content`，
那條路徑需要舊版逐類別的 `prompts` 鍵，`data/animals` 下沒有 `prompts.yaml`。

**沒有替動物寫 `prompts.yaml`。** 那份檔案要寫的是攻擊端的編輯指令與防禦方
選定的 `c_a`，屬於協定層的決定，不由這一批的排程順手定下來。要跑遮罩時先補
那份檔案，並修 `make_masks.py` 對新格式的相容性。
