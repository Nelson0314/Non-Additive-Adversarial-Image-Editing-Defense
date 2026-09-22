# 生成式載體文獻查證

依據 [BRIEF.md](BRIEF.md) 第四節；「編輯免疫」指公開防禦圖後，使固定受害編輯器的後續編輯受阻，不包含分類誤判或防止個人化訓練。
「生成式載體」依求解變數與輸出流程判定：防禦圖由生成／重繪程序直接產生；僅用生成圖作損失目標，或以網路產生受限的加性擾動，另列說明。
任何輸出都可代數地寫成原圖加差值，因此不能只用是否寫成 x+δ 判斷載體種類。
下表以論文原文、官方 proceedings、作者公開稿及出版社資料交叉查證；arXiv-only 條目不冒稱正式會議論文。作者超過三人以「等」節錄。

| 出處（作者、題名、會議／期刊、年、識別碼） | 載體 | 求解變數 | 約束／保真方式 | 威脅模型 | 與本任務的關係 |
|---|---|---|---|---|---|
| Yang Song、Rui Shu、Nate Kushman、Stefano Ermon；Constructing Unrestricted Adversarial Examples with Generative Models；NeurIPS 2018；[arXiv:1805.07894](https://arxiv.org/abs/1805.07894) | class-conditional AC-GAN 從頭合成 | GAN latent；另有疊加小擾動的擴充 | latent 搜尋範圍、auxiliary class 一致性、人類標籤確認；不要求貼近某張原圖 | 分類器，白箱及轉移 | 奠定生成樣本作為 unrestricted attack 的形式，但不能保證原人像身分。 |
| Anand Bhattad 等；Unrestricted Adversarial Examples via Semantic Manipulation；ICLR 2020；[arXiv:1904.06347](https://arxiv.org/abs/1904.06347) | cAdv 色彩化、tAdv 紋理轉換 | 色彩化控制／紋理合成影像 | 色彩化 prior、texture/content 結構；不以小 pixel Lp 為共同預算 | 分類器、image captioning | 證明可見色彩與紋理可承載攻擊；抗 JPEG 的分類結果不能換算成本專案的編輯保留率。 |
| Chenlin Meng 等；SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations；ICLR 2022；[arXiv:2108.01073](https://arxiv.org/abs/2108.01073) | 對輸入加噪再去噪重繪 | 原方法無對抗最佳化；輸入 guide、起始噪聲尺度 | 噪聲強度調節 realism–faithfulness；不是逐圖保真證書 | 一般生成／編輯，非攻擊 | 支持載體構造，不支持「低 strength 必保身分」或「樣本必抗淨化」。 |
| Haotian Xue、Alexandre Araujo、Bin Hu、Yongxin Chen；Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability（Diff-PGD）；NeurIPS 2023；[arXiv:2305.16494](https://arxiv.org/abs/2305.16494) | SDEdit 去噪結果；style 分支先風格化再生成對抗圖 | SDEdit 輸入影像的 PGD 更新，非 prompt embedding | 基本型為 pixel L∞ 投影；style 型以風格化起點、區域 mask 與生成 prior 保持外觀 | 分類器，白箱／轉移／淨化後分類 | 最接近「SDEdit 作對抗載體」；須區分原文回傳的擾動輸入與去噪輸出，style 分支交付後者。 |
| Jianqi Chen 等；Diffusion Models for Imperceptible and Transferable Adversarial Attack（DiffAttack）；IEEE TPAMI，2024 online／2025 卷期；[arXiv:2305.08192](https://arxiv.org/abs/2305.08192)、[DOI:10.1109/TPAMI.2024.3480519](https://doi.org/10.1109/TPAMI.2024.3480519) | Stable Diffusion 反演後的 latent 重建 | 中間 latent；cross-attention distraction 為損失 | self-attention 保持、限制反演深度，非共同 pixel Lp 預算 | 分類器白箱代理、黑箱轉移 | 提供結構保持方法；其 arXiv v1 Appendix C 的 text-embedding 攻擊轉移性偏弱，是草案的重要反例。 |
| Xuelong Dai、Kaisheng Liang、Bin Xiao；AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models；ECCV 2024；[arXiv:2307.12499](https://arxiv.org/abs/2307.12499) | diffusion 直接合成 unrestricted 樣本 | 反向取樣狀態，由分類梯度作 adversarial guidance | 生成 prior 與類別條件；不要求維持指定原圖 | 分類器 | 說明逐步 guidance 可避免保留整條最佳化圖，但不等價於保人像的 img2img 免疫。 |
| Jiang Liu 等；DiffProtect: Generative adversarial examples using diffusion models for facial privacy protection；Pattern Recognition 173，2026；[DOI:10.1016/j.patcog.2025.112780](https://doi.org/10.1016/j.patcog.2025.112780)；前身 [arXiv:2305.13625](https://arxiv.org/abs/2305.13625) | diffusion autoencoder 解碼人臉 | semantic latent code；固定 stochastic code | semantic code 的 L∞ 範圍、face parsing 一致性 | 人臉辨識，targeted impersonation／代理轉移 | 直接生成保護人像且有單步近似加速，但防護 FR 不是阻止後續編輯。 |
| Yuhao Sun、Lingyun Yu、Hongtao Xie 等；DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection；CVPR 2024；[arXiv:2405.09882](https://arxiv.org/abs/2405.09882) | diffusion 生成參考妝容的人臉 | makeup removal／adversarial makeup transfer 的 diffusion 模型微調參數 | CLIP 妝容方向、區域 histogram、LPIPS、L1、非妝容資訊保持 | 人臉辨識，ensemble 代理與黑箱 | 可見、生成式妝容確實作為防護載體，但不是 frozen editor＋embedding，也未建立 editing immunisation。 |
| Jiang Liu、Chen Wei、Yuxiang Guo 等；Instruct2Attack: Language-Guided Semantic Adversarial Attacks；arXiv 預印本，2023；[arXiv:2311.15551](https://arxiv.org/abs/2311.15551) | InstructPix2Pix 的語意／風格編輯結果 | 影像與文字 guidance 的可學習 latent-shaped factors α、β | LPIPS bound、factors 投影至 [0,1]、guidance 調節 | 分類器，白箱與黑箱轉移 | 使用生成編輯器當載體，受害者仍是分類器；不能把其可見編輯指令帶入本任務求解端。 |
| Yasamin Medghalchi 等；Prompt2Perturb (P2P): Text-Guided Diffusion-Based Adversarial Attack on Breast Ultrasound Images；CVPR 2025；預印本題名為 Attacks；[arXiv:2412.09910](https://arxiv.org/abs/2412.09910) | text-guided latent diffusion 重建超音波影像 | prompt/text embeddings；生成模型固定 | 分類損失結合 noise-prediction 一致性、相似度選圖 | 醫療影像分類器 | 提供 embedding 求解實例；醫療影像的相似性結果與文中 timestep 用語不能直接當 SD 人像末 k 步設定。 |
| Yibo Wang、Yong Zhou、Bing Liu、Rui Yao；Style-controllable adversarial example generation via image editing and prompt embedding optimization（StylePromptAdv／SPA）；Neurocomputing，2026；[DOI:10.1016/j.neucom.2026.134591](https://doi.org/10.1016/j.neucom.2026.134591) | frozen text-guided diffusion editor 的風格重繪 | adversarial prompt embedding | 公開摘要／章節摘錄列 Perceptual Trajectory Alignment Loss、prompt consistency loss | ImageNet 分類器 unrestricted attack | 載體與變數最貼近草案；全文未取得，實際 backbone、SDEdit 排程、梯度截斷、顯存與超參數均「未查證」。 |
| Shawn Shan、Jenna Cryan、Emily Wenger 等；Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models；USENIX Security 2023；[arXiv:2302.04222](https://arxiv.org/abs/2302.04222) | 原圖加 style cloak；SD 風格轉換圖只當 feature target | pixel cloak δ | LPIPS perceptual budget／penalty | 個人化微調後的風格模仿 | 不應誤列為直接交付風格轉換圖的編輯免疫；其生成 target 與防禦輸出是兩張不同的圖。 |
| Qiuyu Tang、Joshua Krinsky、Aparna Bharati；StyleProtect: Safeguarding Artistic Identity in Finetuned Diffusion Models；CVPR Workshops（APAI），2026；[arXiv:2509.13711](https://arxiv.org/abs/2509.13711) | 原圖加 adversarial perturbation | pixel 擾動；代理適應僅更新選定 cross-attention layers | 擾動 budget、視覺不可察性量測 | 個人化 diffusion 的風格模仿 | 名稱含 style 仍不是生成式風格濾鏡；其訓練威脅不同於固定編輯器。 |
| Haotian Xue、Chumeng Liang、Xiaoyu Wu、Yongxin Chen；Toward effective protection against diffusion based mimicry through score distillation；ICLR 2024；[arXiv:2311.12832](https://arxiv.org/abs/2311.12832) | pixel 加性保護擾動，非生成式載體 | pixel perturbation；SDS 近似梯度 | 小幅擾動預算；研究 semantic loss 不同方向 | diffusion 編輯／mimicry、個人化相關場景 | 發現 encoder 脆弱性並以 SDS 節省 denoiser 反傳；不能據此宣稱 SDS 消除本草案的生成鏈 Jacobian。 |
| Ron Mokady、Amir Hertz、Kfir Aberman 等；Null-text Inversion for Editing Real Images using Guided Diffusion Models；CVPR 2023；[arXiv:2211.09794](https://arxiv.org/abs/2211.09794) | SD 反演與重繪 | 每個 timestep 的 unconditional embedding；固定 conditional embedding／權重 | 對 pivotal inversion trajectory 的重建誤差 | 一般編輯，非免疫 | 說明 embedding 可最佳化，但其目的是重建且變數是 null-text，不能當作 style embedding 攻擊的有效性證據。 |
| Ben Poole、Ajay Jain、Jonathan T. Barron、Ben Mildenhall；DreamFusion: Text-to-3D using 2D Diffusion；ICLR 2023；[arXiv:2209.14988](https://arxiv.org/abs/2209.14988) | NeRF rendering，非防禦圖 | 3D 場景參數，以 SDS 提供梯度 | diffusion prior 與幾何／rendering 正則 | 一般生成，非攻擊 | SDS 省略 score network Jacobian；仍須穿過 renderer／本草案的 SDEdit 載體，且反轉 SDS 不保證自然。 |
| Haotian Xue、Yongxin Chen；Pixel is a Barrier: Diffusion Models Are More Adversarially Robust Than We Think；arXiv 預印本，2024；[arXiv:2404.13320](https://arxiv.org/abs/2404.13320) | 比較 LDM／pixel diffusion 對輸入擾動的反應 | 輸入影像的白箱梯度攻擊 | 各攻擊的擾動預算；非風格生成設計 | 編輯／mimicry 防護及其淨化 | 提供防護可被 pixel diffusion 淨化的反證，不能從「用了生成 prior」推出免疫或抗淨化。 |

**具體問題的回答。**

在上述可查證文獻及其相關工作的檢索範圍內，**未找到直接把生成式風格轉換的輸出當成防禦圖，並以固定 image editing 模型的後續編輯失效為目標的已發表實證**；這是限定範圍的檢索結論，不是「不存在」或新穎性的證明。
SPA、Diff-PGD 的 style 分支、Instruct2Attack 的 style 編輯是最接近的生成式載體，但損失與評測終點為分類器；DiffAM／DiffProtect 是生成式隱私防護，但終點為 FR。
Glaze 會呼叫 style transfer，交付物卻是貼近原作品的 cloak；StyleProtect 的 style 指受保護的藝術風格，兩者均不能充當所問先例。
「用網路產生防禦圖」則有直接先例：主表的 **DiffVax**（Tarik Can Ozden 等，ICLR 2026；[arXiv:2411.17957](https://arxiv.org/abs/2411.17957)）訓練 image immunizer 產生加性擾動，推論只需一次 forward；不是用現成 SD 做可見風格重繪。
主表十二條件不另製文獻表；以「可學習網路產生編輯免疫圖」而言 **diffvax 最近**，以 encoder 損失而言則接近 **PhotoGuard**（Hadi Salman 等，ICML 2023；[arXiv:2302.06588](https://arxiv.org/abs/2302.06588)）。草案的最大化 encoder 距離也不等同 PhotoGuard 原始 targeted encoder objective。

**查證界線與可轉用的結論。**

SPA 的 DOI／題名／期刊可由 [本地 Elsevier metadata](paper/neucom_134591.json) 核對；作者、classifier 設定及損失名稱另有 [出版社摘要與章節摘錄](https://www.sciencedirect.com/science/article/pii/S0925231226019892) 支持，沒有把摘要推測成已讀全文。
DiffProtect 的 journal 版採 2026 卷期，DOI 內的 2025 不是卷期年份；[作者機構紀錄](https://pure.johnshopkins.edu/en/publications/diffprotect-generative-adversarial-examples-using-diffusion-model/) 可核對作者與 DOI。
DiffAttack 的 [arXiv v1 Appendix C](https://arxiv.org/html/2305.08192v1) 明載 text embedding 轉移性問題；這是特定分類設定的觀察，不能擴張成所有 embedding 攻擊必敗。
Diff-PGD 的 [§6、Appendix D](https://arxiv.org/html/2305.16494v2) 有 Jacobian 常數近似及 A6000 成本；其輸入影像 Jacobian 近似不能直接充當 77×768 embedding 到 RGB 的 Jacobian，也不能直接套成 RTX 3090、SD v1.5、512² 的實測時間。
研究缺口是同時驗證「生成式保真」「未知編輯指令下的編輯端增量」「主表淨化保留率」，不是只證明 embedding 能讓某個代理分數上升；設計與成本推算見 [DESIGN_REVIEW.md](DESIGN_REVIEW.md)。
