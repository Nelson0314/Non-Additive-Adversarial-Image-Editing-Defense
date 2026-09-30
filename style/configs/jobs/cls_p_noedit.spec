# 輪 cls_p_noedit：論文完全移植（classifier 損失，20 步），p_noedit；ref 為 lr 0 的對照。
# 原遠端工作清單 specs/style_prompt_cls_p_noedit.txt；格式：名稱 影像（多張以 + 連接） 風格 其餘參數。
# 用法（style 專案根）：bash scripts/run_style_prompt_jobs.sh cls_p_noedit configs/jobs/cls_p_noedit.spec
ref man_01+woman_02 p_noedit --objective classifier --sampler ddpm --carrier prompt --prompt-scope full --a-perc 25 --a-prompt 1 --a-adv 1 --init-rms 0 --delta-rms-cap 1e9 --a-id 0 --col-cap 0 --struct-cap 0 --patience 1000000 --max-decays 0 --select last --snapshot-every 0 --val-every 100 --lr 0 --updates 1
cls_m01 man_01 p_noedit --objective classifier --sampler ddpm --carrier prompt --prompt-scope full --a-perc 25 --a-prompt 1 --a-adv 1 --init-rms 0 --delta-rms-cap 1e9 --a-id 0 --col-cap 0 --struct-cap 0 --patience 1000000 --max-decays 0 --select last --snapshot-every 0 --val-every 100 --lr 0.1 --updates 20
cls_w02 woman_02 p_noedit --objective classifier --sampler ddpm --carrier prompt --prompt-scope full --a-perc 25 --a-prompt 1 --a-adv 1 --init-rms 0 --delta-rms-cap 1e9 --a-id 0 --col-cap 0 --struct-cap 0 --patience 1000000 --max-decays 0 --select last --snapshot-every 0 --val-every 100 --lr 0.1 --updates 20
