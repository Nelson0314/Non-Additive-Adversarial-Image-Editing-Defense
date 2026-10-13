# 文獻的評估集規模

| 文獻 | 年份／出處 | 評估集 | 來源 |
|---|---|---|---|
| PhotoGuard | ICML 2023 | 60 個編輯（"we generate 60 different edits of a variety of images using different prompts"） | https://arxiv.org/html/2302.06588 §4.2 |
| EditShield | arXiv 2311.12066 | 2,000 張 IPr2Pr ＋ 1,000 張 MagicBrush | https://arxiv.org/html/2311.12066 |
| DAYN (Distraction is All You Need) | CVPR 2024 | 150 張生成影像、3 類、每類 2 個 prompt | docs/reference/AUDIT_DAYN.md:109 |
| DiffusionGuard | ICLR 2025, arXiv 2410.05694 | InpaintGuardBench：42 張 × 5 遮罩 × 10 prompt ＝ 2,100 個編輯 | https://arxiv.org/html/2410.05694 |
| FaceLock | CVPR 2025, arXiv 2411.16832 | CelebA-HQ 2,000 張人像；25 條指令（三類） | https://arxiv.org/html/2411.16832 |
| DiffVax | ICLR 2026, arXiv 2411.17957 | CCP 875 張：800 訓練（含 75 seen）、75 unseen 測試 | https://arxiv.org/html/2411.17957v2 |
| DCT-Shield | arXiv 2504.17894 | OmniEdit 150 張；inpainting 另 56 張 | https://arxiv.org/html/2504.17894 |
| DANP | — | InstructPix2Pix-clip-filtered 抽 200 張 | docs/reference/AUDIT_DANP.md:162 |
| SIFM（Towards Transferable Defense Against Malicious Image Edits） | arXiv 2512.14341 | 自建 100 張（人像 35／風景 35／畫作 30） | https://arxiv.org/html/2512.14341；AUDIT_SIFM.md:105 |
| DIA | — | PIE-Bench 700 張 | docs/reference/AUDIT_PURIFIERS.md:736 |
| PromptFlare | arXiv 2508.16217 | EditBench 240 張 | docs/reference/AUDIT_PROMPTFLARE_PHOTOGUARD.md:348 |
| Evaluating Adversarial Protections for Diffusion Personalization | arXiv 2507.03953 | VGGFace2 50 身分 × 8 張、WikiArt 50 藝術家 × 8 張 | docs/reference/BASELINE_ALIGNMENT.md |
| DeContext | arXiv 2512.16625 | VGGFace2 與 CelebA-HQ 各 50 個身分；2–6 個 prompt | https://arxiv.org/html/2512.16625 |
| Universal Image Immunization via Semantic Injection | arXiv 2602.14679 | 500 張生成影像（10 類 × 50），每張 2 個 prompt | https://arxiv.org/html/2602.14679 |
| VETO | arXiv 2607.27292 | EditBench、AnyEdit、VetoBench 各 300 個樣本 | https://arxiv.org/html/2607.27292 |
| NullEdit | arXiv 2608.10870 | CelebA-HQ 與 VGGFace2 各 50 個身分（每身分 1 張），每張 12 條指令 | https://arxiv.org/html/2608.10870 |

影像數由小到大：42、60、75、100、100（50×2）、100（50×2）、150、150、200、240、500、700、800（400×2）、900（300×3）、2,000、3,000。
16 篇的中位數為 150–200；9 篇在 100–300 之間。


查證日期 2026-10-06。影像數取各篇主要量化表所用的來源影像；同一張影像搭配多個遮罩或指令時，編輯數列於「評估集」欄。
達到各規模所需的 GPU 時間見 `docs/PEXELS_PORTRAITS.md`「GPU 時間」。
