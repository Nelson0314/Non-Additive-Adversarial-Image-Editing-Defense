# 已刪除的暫時性嘗試

暫時性嘗試放在不入版控的 `trials/<名稱>/`（`bash vendor/scripts/run_trial_lifecycle.sh new <名稱>`）。採用時把需要的程式、設定與結果搬入 `src/`、`configs/`、`results/` 等目錄並提交，在 `trials/<名稱>/PROMOTED` 逐行列出目的檔，再執行 `run_trial_lifecycle.sh promote <名稱>`：目的檔須已提交，升格紀錄（目的檔、SHA-256、commit）寫入本檔末的「升格紀錄」。不採用時先在下表寫一列並提交（五欄皆須填寫，未量測寫「未量測」；結論來源不可指向 trial 本身），再執行 `run_trial_lifecycle.sh drop <名稱>`。兩者都經 `TRIAL_REMOTE`、`TRIAL_REMOTE_ROOT` 一併刪除遠端副本；只處理本機時明確給 `--local-only`。

| 名稱 | 試了什麼 | 設定 | 關鍵數字 | 結論來源 |
|---|---|---|---|---|
| `color_normalization_purification` | 未完成：四種色彩處理（灰世界、自動色階、逐通道直方圖均衡、灰階）施於防禦圖後送 ip2p 的淨化實驗；只執行了處理後輸入的產生（CPU，2,285 張 PNG，中途停止），未編輯、未取卡 | 規劃：crosseditor100 的 100 張、ip2p 7.5／1.8／50 步、6 條件（＋SIFM） | 未量測 | 使用者於 2026-11-09 經 executor 取消 |
| `isr_prompt_and_guidance` | 未執行：ISR 偏低來源的 pilot（指令、ip2p image guidance、VIBE-DistilledCFG、原生預算 PhotoGuard encoder），只建立目錄與程式草稿 | 規劃：isr50 前 10 張、5 條件、8 句、ip2p 7.5／1.8 與 7.5／1.2、VIBE-DistilledCFG 20 步、淨化 none／jpeg50／vae_trans | 未量測 | 使用者於 2026-11-08 經 executor 取消，未啟動任何計算 |

## 升格紀錄

| 名稱 | 目的檔與 SHA-256 | commit |
|---|---|---|
| `non_sd_editor_speed` | `scripts/run_nonsd_edits.py 22d8d0fef0e965ba04b433014b48bf86b93da581908779b35724c1da03d47cf3`<br>`scripts/measure_nonsd_editor_speed.py 9e307bbef0db0cd1575c5defc8bf0a670267ac417194134f90aef7e6dab9a4b8`<br>`scripts/evaluate_pexels_nonsd_editor.sh 3b0bda9e525185f4e1763637ac0a2987111221cd494eab0d1a638894305395c1`<br>`results/pexels_portraits/nonsd_editor_speed/speed_klein.json 3ec503b47b181fd7a8efd583b6f246eca60bae59c7dbbd2efe732c89f66ced29`<br>`results/pexels_portraits/nonsd_editor_speed/speed_klein_base.json 9d4e2413adcbfb12731566555b9ba74052fc470422e8bee00836c76d88803de2`<br>`results/pexels_portraits/nonsd_editor_speed/speed_vibe.json 8e21bfd11c835b23454a5f038a8298d5ca8e571352dca8e01919f8fedd39fcc9`<br>`results/pexels_portraits/nonsd_editor_speed/speed_vibe_distilled.json e67c6d439562f69fdf2a5c911585138011b9ec3ffc0f3bad4805ab652190fc70`<br> | `ab567f98c4b6df9dfd65a5735451f11d0ea1951f` |
