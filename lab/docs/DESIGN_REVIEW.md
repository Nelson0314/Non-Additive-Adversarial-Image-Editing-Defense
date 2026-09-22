# style_edit 設計審查

依 [BRIEF.md](BRIEF.md)、[COLOUR_LINE.md](../../COLOUR_LINE.md) 與指定封裝作靜態審查；未執行 GPU。以下顯存／時間為工程估算，**不是 RTX 3090 實測**；文獻出處集中於 [LITERATURE.md](LITERATURE.md)。
建議保留 style_random／style_opt 配對設計；生成器凍結，以受限風格 embedding 搜尋代替自由的 77×768 張量。生成自然性、抗淨化與編輯免疫都仍需影像及編輯端證據。

**(a) 梯度、可行步數與成本：完整梯度應穿過生成鏈；24 GB 上不必直接放棄全鏈。**

若宣稱求解原式對 e_s 的真梯度，必須包含所有使用 e_s 的去噪步，以及 decoder→受害 encoder／UNet 的路徑；固定 x 的初次編碼與初始噪聲可預先計算。模型權重設 requires_grad=False 不會切斷對輸入／embedding 的梯度。
以 N 表示實際 scheduler updates，CFG 每步有兩次 batch=1 UNet forward。diffusers 的名義 50 步在 s=0.2–0.45、order=1 時約為 N=10–22；多階 scheduler 須記錄實際 calls。[官方 img2img 說明](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)
本地 [sd.py](../../anti-purification/src/models/sd.py) 的 sdedit 則先設 t0≈1000s，再完整執行 num_steps 次，故 num_steps=50 就是 N=50；其時間格點、VAE mean 與終點也不能假定和官方 pipeline 等價。
封裝可反傳，_eps 已用 use_reentrant=False checkpoint；必須明給 use_ckpt=True、vae_ckpt=True、CFG 與空文字 embedding，預設 guidance_scale=1 不是草案預期的文字引導。空文字 embedding 不是全零。
[diffusers v0.39.0 原始碼](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py) 的 pipeline.__call__ 有 @torch.no_grad；僅傳 prompt_embeds 或在外層 enable_grad 無法恢復內層計算圖，應用可微分封裝／明確去噪迴圈。
本地 [ip2p.py](../../anti-purification/src/models/ip2p.py) 的 edit_differentiable 已支援 grad_steps 截斷，卻要求非空編輯 instruction；可借用梯度處理方式，不能因此把 edits.* 接入求解端。
成本假設：batch=1、512²、UNet fp16／bf16、scheduler 與 embedding optimizer fp32、memory-efficient attention、CFG 分支循序、每次僅一個受害 UNet 求梯度；保真網路小型且凍結，不保存所有 attention maps。
單一 SD1.5 UNet 約 0.86B 參數，半精度約 1.6 GiB；生成器＋單一受害器＋VAE／文字／保真模組約 4–6 GiB，runtime／allocator 約 1–2 GiB。77×768=59,136 個變數的 fp32 Adam 含梯度與兩個 moments 約 0.90 MiB，瓶頸是 activations。
無 checkpoint 時按每個 UNet branch 暫存約 1–3 GiB 估算，N=10、雙 CFG 即約 25–70 GiB；24 GB 僅適合約 1–3 步的保守規模，不能把 10–22 步當普通 inference 的顯存。
完整 UNet checkpoint 使各步只留 latent／embedding 邊界，單張 fp32 latent 4×64×64 僅 64 KiB；反傳重算的單個 UNet／VAE activation 峰值約 5–12 GiB。含尾端代理，規劃總峰值約 12–20 GiB，須留餘裕；不能把此區間當可保證的上限。

| 方案 | 實際 N／反傳步數 | 估計峰值 | 每次最佳化更新時間 |
|---|---|---|---|
| 全鏈 checkpoint，建議起點 | N=10–20／全部 | 約 12–20 GiB，可合理規劃於 24 GB | 約 9–42 秒 |
| 全鏈 checkpoint，長鏈 | N=50／全部 | 約 13–22 GiB，理論可行但餘裕較小 | 約 35–90 秒 |
| 截斷反傳 | N=20／最後 k=4 | 約 11–18 GiB | 約 8–23 秒；前 16 步仍須 forward |

時間模型：設單分支 forward f=0.08–0.20 秒（估算假設），checkpoint 重算＋backward 約 3f，VAE／單一代理／保真尾端 C=3–10 秒，則 T≈(2N+6k)f+C；實作或硬體若較慢，依公式同比放大。全鏈 N=20、100 次更新約 25–70 分／圖，8 張約 3.3–9.3 GPU 小時，未含評測；若沿用 900 次更新，約 30–84 GPU 小時。
以上採逐受害模型／交替抽樣求 ∂L/∂x_def 並釋放其圖，再對生成器做 VJP；若選擇同噪聲重算生成器以降低並存峰值，每次另加約 2Nf。不可同時堆疊兩條 50 步受害編輯鏈，不能把載體取樣成本當整個實驗成本。
建議生成器起點為 N=20 實際步、s=0.30、CFG=7.5、固定 r，兩臂一致；受害端仍為原協定 50 步。沒有 GPU profile，實際峰值／吞吐與可容納最大 N 均「未查證」。
若資源不足，首選最後 k=4 步反傳：每次以更新後 e_s 重算前綴、在切點 detach，後綴保留 e_s。這是有偏梯度，忽略 e_s 對前綴的作用；固定快取舊前綴、卻宣稱最佳化整鏈會造成求解與交付圖不一致。
SDS 只省略受害 score network 的 Jacobian，不會自動消除 e_s→SDEdit→x_def 的生成鏈；反向 SDS 也不是 ε-MSE 的精確梯度，不能作無成本替代。直接最佳化 z 再 VAE decode 雖便宜，卻失去 diffusion prior，容易重回離流形影像；兩者均不作首選。
若無法維持可微鏈，較一致的退路是固定風格字典上作離散／低維無梯度搜尋，每個候選仍完整 SDEdit；代價是多次 forward 與搜尋能力下降，且仍不得用正式編輯指令選解。

**(b) 目標選擇：選受害 UNet 的條件 ε-prediction 誤差作探索目標；不選單純 VAE 距離最大化。**

理由是 ‖E(x_def)−E(x)‖² 很容易只量到合法風格差異；encoder 正確表達新風格也會使它上升，與編輯失效沒有必然關係。此 untargeted 距離亦不等同 PhotoGuard 的 targeted encoder attack。
建議 L_def=E_{v,t,ε}[MSE(ε_v(q_t(E_v(x_def)),t,c_v(x_def),content),ε)]，q_t(z)=√ᾱ_t z+√(1−ᾱ_t)ε；v 等權取 ip2p／inpaint，t 依各受害 scheduler 分層抽樣，均值按 latent 元素數正規化。
c_v 必須走真正條件入口：ip2p 是 8 通道、附未縮放 posterior mean；inpaint 是 9 通道、附遮罩與「遮罩後影像再編碼」latent。不可用 SD1.5 的四通道空 prompt UNet 冒充兩個受害器，或以補零通道替代條件。
content 僅為 man／woman，並非真實編輯 instruction；空 prompt 誤差只作診斷。上述一步 conditional denoising loss 仍不是 s_t=7.5／s_i=1.8 的完整 CFG 編輯目標，這個 mismatch 必須明列。
每張以相同 t、ε 同時計算原圖／style_random／style_opt 的代理誤差並逐項報告；固定參考值的相減只改善解讀，不改變梯度，更不能排除風格／紋理本身導致高 MSE。
**不能保證避開同一個坑。** 專案四代理推動 0.003–3.8 倍仍未破壞編輯，而且已有 sds 項；改名 UNet loss 沒有新證據。可能改善的是內容相依載體的可達方向及正確受害條件，而非「代理換了所以有效」。
Xue 等 [arXiv:2311.12832](https://arxiv.org/abs/2311.12832) 反而指出 encoder 的脆弱性；這不支持 UNet 必較易攻破。選 UNet 是為減少把正常 latent 位移當成果，須接受優化不到免疫效果的可能。
只以固定完整 64 格的編輯結果、逐格圖與配對差陳述增量，並列輸入失真；不得按 edits.* 的效果調 e_s、s、checkpoint 或 λ，也不新增成功門檻。style_opt 代理勝出但編輯端與 style_random 打平，必須照報。

**(c) 保真：自然度應進參數化的原則仍成立，但生成器只提供 prior，沒有自然影像流形的硬保證。**

自由最佳化 59,136 維 contextual embedding 可離開 CLIP 訓練分布；凍結 UNet、低 strength、LPIPS 小或 DISTS 小，都不能保證膚色、五官與身分自然。最大化去噪誤差更可能與生成 prior 對抗。
採固定少量攝影風格字典，把條件限制為 e(a)=Σ_j a_j CLIP(content＋style_j)、a_j≥0、Σ_j a_j=1；s、CFG、r 固定，先不加入任意 embedding residual。此凸包限制降低越界自由度，但混合 embedding 仍不是自然度證書。
此處 style_j 是 BRIEF 明示允許的防禦載體控制，影像相關文字只取 content；風格字典不得來自 edits.* 或推測攻擊指令。若要更強的可解釋性，改選離散字典候選；代價是搜尋容量較低。
結構優先由原圖的同噪聲參考軌跡／self-attention 結構保持或保護區 latent anchoring 進生成程序；兩臂須共用。既有 subject mask 不是 face mask，不可假定已保臉；錨定整個主體又可能把所需防禦自由度全部鎖死。
保真項選 **DISTS 對原圖的結構／紋理距離**作輔助控制，並報 LPIPS、PSNR、全圖／主體 ΔE00 與尾端分布；DISTS 對紋理較容忍，仍會漏掉身分替換，不能取代逐圖檢視。[Ding 等，arXiv:2004.07728](https://arxiv.org/abs/2004.07728)
保留既有指標但不另宣告 LPIPS／ΔE00／DISTS 的自然度通過線；λ 或 s 掃描只呈現失真–位移資料及全部樣本，不以單一分數篩出「自然」輸出。參數化縮小可行族，也不保證存在同時自然且免疫的解。

**(d) 最可能的靜默失效與對應讀法。**

| 失效點 | 具體診斷／修正 |
|---|---|
| 重繪退化為全域色調；高 LPIPS 只是風格被帶進輸出 | 與 colour_curve_ours、style_random 並列輸入失真–編輯位移；額外作 tone-curve 擬合殘差與灰階／局部紋理診斷，主表 LPIPS 定義不變。 |
| 更難編輯的臉其實已被生成器換人 | 先看原圖→防禦圖的五官、輪廓及身分；s=0.2–0.45 都不保證安全。換人樣本保留並註記，不能靠重新選分母或刪圖改善讀數。 |
| embedding 漂移、CFG 放大及 decoder clamp 製造怪圖 | 報條件範數、字典權重、clamp 飽和比例；檢查原圖／random／opt。77×768 自由座標或軟指標達標皆非自然證明。 |
| 生成的顆粒、局部紋理被 JPEG／blur 洗掉 | tone curve 的全域色彩機制不會自動轉移到局部紋理；逐算子報絕對位移與比值。DiffPure 不在固定算子集合，不能聲稱已驗證其效果。 |
| 梯度斷裂或截斷只推動末端高頻細節 | 檢查 no_grad／detach／PIL／8-bit 往返、e_s 梯度非零；固定噪聲作方向有限差分與小 N 全鏈對照。量化後再評估交付 PNG，不能只報浮點候選。 |
| 原圖 VAE 重建誤差、seed／scheduler 差異被算成免疫 | 兩臂用同一 VAE、posterior 規則、scheduler、噪聲 tensor、s 與 CFG；random 必為 opt 的同起點未最佳化輸出，另列 VAE reconstruction 誤差。 |
| inpaint 保留區原封攜帶風格，編輯卻照常成功 | 同時看主體內／外位移及逐格指令執行；保留區差異大不等於白色重繪區被阻止。 |
| 代理過擬合、依正式 64 格選圖造成指令洩漏 | 防禦設定與預算固定後才讀 edits.* 評測，報告所有影像與指令；不以某次代理或正式編輯分數挑風格／種子。 |
| 淨化比較兩側不對稱、identity 混入均值、分母近零 | 用編輯(P(x)) 對編輯(P(x_def))；重用主表相同 P 的未防禦控制，幾何遮罩同步變換；逐項報比值、分子分母及既有 net_gain，不藏低分母樣本。 |

**評測與對照的必要細節。**

style_random 的字典起點／風格分派須預先固定；style_opt 從相同起點求解，保持 s、r、步數、結構保持方式一致。若增加強度／風格設定，每個設定均要有完整配對，不挑最佳 8 張拼成一臂。
沿用 8 張 512² 人像、每場景 4 指令、共 64 格；ip2p s_t=7.5、s_i=1.8，inpaint guidance=7.5、dilate 4、白＝重繪；受害端種子 20260812、50 步；複用 ip2p_si18／inpaint_undefended 與淨化未防禦控制，不重跑或更換分母。
[lab/code/purify_run.py](../code/purify_run.py) 實列 identity、crop_resize0.1、jpeg30/50/80、blur1/blur2、rotate15，共 **8 個標籤＝identity＋7 道非恆等處理**；五道非幾何處理是 JPEG 三道與 blur 兩道，identity 單列、兩道幾何分列。
故每臂保留未淨化 64 格及七道非恆等的 448 格；若照程式輸出全部標籤，淨化表為 8×64=512 格、淨化圖 8×8=64 張。這是計數釐清，不刪改既有算子；兩臂排除 identity 重複後至少 1,024 次受害編輯，成本不可忽略。
