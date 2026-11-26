# Code-vs-paper audit checklist (method, Sec. 3 of 3_method.tex)

Each item: paper statement -> where in code (file:line) -> match / mismatch.

## 3.2 LUT parameterization
M1  T: [0,1]^3 -> [0,1]^3; RGB normalized to [0,1] before mapping.
M2  One LUT per image (no sharing across images).
M3  K x K x K lattice, uniformly sampled nodes; K = 33 (Sec. 4 / supp).
M4  Trilinear interpolation from 8 neighboring nodes (not tetrahedral, not nearest).
M5  Same mapping applied to every pixel; no spatial dependence (no position input, no per-region LUT).

## 3.3 Geometry
M6  Gray axis g = (1,1,1)/sqrt(3).
M7  Spherical coords c = ||c|| u(theta, phi); theta = polar angle from g; phi = azimuth around g.
    Check: origin at black point (0,0,0); azimuth reference direction defined consistently.
M8  T(c) = clip_[0,1]( ||c|| u(theta+dtheta(c), phi+dphi(c)) ); ||c|| preserved before clipping; clip is the only post step.
M9  Theta in R^{3 x K^3}: learnable vector field in RGB Euclidean space (3 components per node).
M10 V = G_sigma * Theta, separable 3D Gaussian filter; sigma = 4 lattice units (supp / Sec. 4). Check kernel radius/truncation and boundary padding mode (paper silent -> report what code does).
M11 Initialization: Theta small random values; supp: 0.002 * N(0,1) "around the identity mapping"; check seed (supp: solver seed 0).
M12 No additional smoothness regularization loss.
M13 v_theta = <V, e_theta>, v_phi = <V, e_phi>; e_theta, e_phi orthogonal unit tangent vectors at each lattice node.
M14 s_a(t) = arctan( tan(a) * tanh( t / tan(a) ) ); maps R -> (-a, a).
M15 dtheta = s_thetamax(v_theta); dphi = s_phimax(v_phi / r), r = sin(theta + dtheta).
    Check order: r uses the UPDATED polar angle theta+dtheta.
M16 theta_max = 25 deg, phi_max = 7.5 deg (Sec. 4 / supp).
M17 On the gray axis (theta = 0): dphi = 0. Check epsilon used and behaviour near the axis and at black (||c|| = 0).

## 3.4 Objective
M18 Loss = (1/n) || mu_E(T(x)) - mu_E(x) ||_2^2; n = number of latent elements (mean squared error, not sum, not L1).
M19 mu_E = posterior MEAN of the VAE encoder (not a sample, not scaled by 0.18215 unless paper says so). Check quant_conv inclusion and input scaling to [-1,1].
M20 Encoder = InstructPix2Pix VAE (timbrooks/instruct-pix2pix), frozen weights.
M21 Maximize (gradient ascent / minimize negative).
M22 x~ = T_{Theta*}(x): final output applied to the ORIGINAL image with optimized Theta*; which iterate is Theta* (last vs best) -> report.
M23 Optimizer (supp A): Adam, 1800 steps, lr decayed 3e-4 -> 1e-5 (check schedule shape: cosine/linear/exponential), no distortion constraint in optimization, outputs clipped to [0,1].
M24 Output quantization to 8-bit PNG (supp mentions quantized image for LPIPS).
M25 Image resolution used during optimization vs. delivered image (resize? 512?).

## Sec. 4 / supp C (to be filled from 4_experiments.tex and X_suppl.tex)
E1  ip2p: text guidance 7.5, image guidance 1.8, 50 steps, seed 20260812, 4 FaceLock instructions (exact strings).
E2  Purifications: JPEG 80/50/30, Gaussian blur sigma 1/2 (kernel size?), crop-resize 0.1, rotate 15; VAE-Trans settings.
E3  Metrics: piq.LPIPS backend, resize/interpolation, edit LPIPS defined vs undefended edit of the same purification.
E4  Ablation variants A1-A4, B1-B4 exactly as supp A.
E5  Baseline budgets, eps_scale, step sizes as tab:supp_provenance.
E6  SD inpainting: CLIPSeg CIDAS/clipseg-rd64-refined, class-name prompt, threshold 0.30, dilation 4 px, 50 steps, guidance 7.5, strength 1.0, 512x512, composite outside mask.

## Audit results (immulut/lut.py, solve.py, ip2p.py, io.py, alignment.py, configs/immulut.yaml)

- M1-M25: all items match the paper or are covered by the user's decisions (Adam betas and schedule documented in README; loss -w*d/d0 ported; r = sin(theta+dtheta)/cos(dtheta) left unchanged by decision; Gaussian 3-sigma truncation with replicate padding and black-point / gray-axis handling documented).
- Trilinear interpolation: grid_sample on a 5D table with align_corners=True; table axes (b, g, r) match grid coordinates (r, g, b).
- TF32 disabled at import in immulut/ and evaluation/, as in the code that produced the reported results.
- Finding: the output is ||c|| * normalize(d + t * e_phi(d)); adding the azimuthal component increases the polar angle slightly beyond theta + dtheta (sin factor at most sqrt(1 + tan^2(dphi)) <= 1.0087 under the 7.5 degree bound); the azimuth bound holds exactly. Same mechanism as the r definition. Decision: left unchanged.
- face_box raises when no face is detected (all paper inputs are portraits).

Not yet audited: evaluation/, immulut/color_mappings.py, immulut/solve_ablation.py, scripts/, baselines/, final README.md.

## Audit results (baselines/, read-only audit by the baseline session)
Paper = X_suppl.tex Sec. B, Sec. C, Table 7 (tab:supp_provenance).

Match: E5 budgets/steps/norms/factors for PhotoGuard (encoder), Mist, DIA-PT/-R (eps, steps), DCT-Shield/-Y (eps 1, gamma 0.1 linearly decayed, 1000 steps, Q_alg 0.95/0.85, Y only), SIFM (7.65/255, 1/255, 100), DAYN (0.06 on [-1,1] = 0.03, 2/255 on [-1,1] = 1/255, 100), DANP (0.03, 1/255, 100), BlurGuard (16/255, lr 0.06, 150, sigma-weighting 10; matched: 0 and x0.6071); matched factors immunize.py MATCHED_EPS_SCALE = Table 7; step size scaled with eps (immunize.py solve); DCT gamma scaled with eps; calibration anchor 0.3344 on 8-bit PNG with piq.LPIPS (calibrate.py, evaluation.metrics.lpips); solver seed 0 (immunize.py --seed default 0; generate_blurguard_defenses.py:117); 10 gradient repetitions for DAYN/DANP; Mist target text "a painting", fused mode 2 (mist.py:10, :40); E6 model, class-name prompt, threshold 0.30, dilation 4, 50 steps, guidance 7.5, strength 1.0, 512, seed 20260812, composite outside mask (editors/inpaint.py, subject_masks.py); E6 prompts ("A {content} in a hospital / in the gym / in a snowy street / sitting next to a dog"; README lists them); klein/VIBE settings (editors/cross_editor.py:20-21) = Sec. C Editors; anonymity grep clean.

Mismatch:
- A1 DIA step size. Table 7: "1/255" (budgets on [0,1] unless stated). Code attacks/dia.py:101 step_size=1/255 on the [-1,1] range, i.e. 0.5/255 on [0,1] (pgd.py:83). README Baselines table repeats "1/255". The 1/255 value is the official attack_setting.json lr in [-1,1] units.
- A2 PhotoGuard objective. Sec. C: "maximizes the l2 norm of the posterior mean". Code attacks/photoguard.py:54-55 minimizes ||mu_E(x_adv)||_2 (pgd.py:187), as in the official encoder attack with a zero target.
- A3 DCT-Shield objective. Sec. C: "maximizes the l2 norm of the encoder latent". Code attacks/dct_shield.py:46 minimizes ||E(x')||_2 (delta -= sign(g)*eta).
- A4 SIFM objective. Sec. C: "uses a mean squared error on the mid-block output". Code attacks/sifm.py:132 minimizes L_norm - lambda*L_dist, L_norm = L1 of mid-block features, L_dist = MSE, lambda = 0.1 (sifm.py:7).

Paper silent (code behaviour):
- A5 PhotoGuard: step size linearly decayed 0.01 -> 0.0001 over 1000 steps; uniform L-inf init.
- A6 Mist: uniform L-inf init.
- A7 DIA-PT: L1-ball random init; DIA-R no init.
- A8 Victim of all PGD-type baselines is SD v1.4 (CompVis/stable-diffusion-v1-4), fp32, 512; only SIFM states SD v1.4 in the paper. Its VAE encoder equals that of InstructPix2Pix.
- A9 BlurGuard: SAM automatic masks, points_per_side=2; data root needs man/ and woman/.
- A10 Subject mask: max-pool dilation with a 9x9 kernel (radius 4); mask area must lie in [0.01, 0.8] or the run stops; repaint mask binarized at 0.5.
- A11 Inpainting: fp32, safety checker off; raw output saved as __raw.png.
- A12 Cross-editor inputs resized to 512x512 with Lanczos; VIBE seeded with torch.manual_seed, klein with a CUDA generator per edit.
- A13 Calibration procedure (coarse sweep {0.05,0.1,0.25,0.5} on one image, log-log interpolation, 2/255 level for saturated L-inf methods) documented in calibrate.py only.
- A14 Default runs are not bit-identical (--deterministic, already decided).

README: Baselines table = code except A1; the "Loss" column for PhotoGuard/DCT-Shield does not state the direction.

## Audit results (evaluation/, color_mappings.py, solve_ablation.py, scripts/, README.md; read-only audit by the baseline session)
Decided items (Adam betas, cosine schedule, -w*d/d0, r, polar angle, Mist license, --deterministic) not repeated.

Match: E1 timbrooks/instruct-pix2pix, text 7.5, image 1.8, 50 steps, seed 20260812, fp32 (configs/immulut.yaml:17-21, evaluation/edit_ip2p.py:16-21); scheduler = model default (Euler-Ancestral); instruction strings configs/immulut.yaml:22-26. E2 JPEG 80/50/30, blur sigma 1/2 (radius ceil(3 sigma), reflect), crop 10% per side + bicubic resize, rotate 15 bilinear zero fill (evaluation/purify.py:19-55); VAE-Trans encoder+quant_conv fine-tuned, frozen decoder, posterior mean (evaluation/vae_trans.py:13-23); training sigma 0.1, posterior-mean target, MSE, AdamW 1e-4, 6 epochs, batch 2, 512 random crops, seed 0 (scripts/train_vae_trans.py). E3 piq.LPIPS default (evaluation/metrics.py:12); edit LPIPS = LPIPS(edit(P(y)), edit(P(x))) same seed (run_protocol.py:64-72); 8-bit quantization of purified inputs, edits and defended images. E4 A1-A4 options; B1 bisection to +-0.005 on 8-bit images with |h| capped at pi; B2/B3/B4 parameterizations; B2-B4 init 0.002 N(0,1) seed 0 and ImmuLUT objective/schedule; +-0.015 checkpoint rule. Anonymity grep clean.

Mismatch:
- B1 Distortion constraint in B2-B4. Supp A: "No distortion constraint enters the optimization" (all optimized variants) and only checkpoints within +-0.015 are kept. Code color_mappings.py:118-183 solve_matched adds the defense LPIPS interval [t-0.015, t+0.015] to the loss through an augmented-Lagrangian (PHR) term updated every 50 steps (soft upper target t, floor term, rho doubling up to 16); the checkpoint check is in addition. B2-B4 are optimized under a distortion constraint; A1-A4 and ImmuLUT are not.
- B2 Re-runs. Supp A: B2-B4 use the ImmuLUT solver (1800 steps, 3e-4 -> 1e-5). README.md:103-105: B2/B3 portraits without a feasible checkpoint were solved again with steps=3600 lr=0.001 or steps=3600 lr=0.0003 lr_final=0.0003. Not in the paper.

Paper silent (code behaviour):
- B3 B1 search: alpha starts at 0.01, doubles until target or cap; geometric-mean bisection, at most 60 iterations.
- B4 Parameter scaling: B2 (h, s) = 10*delta, B3 (A, b) = 3*delta; B4 table projected to [0, 1] after every step.
- B5 B2-B4 delivery: last iterate if feasible, else last feasible checkpoint within the final 100 steps, else the feasible checkpoint with the lowest loss, else no output.
- B6 B2-B4 objective includes the face-box reference weight w as in ImmuLUT.
- B7 JPEG input cast to uint8 by truncation after *255 (purify.py:22, as in the original code); float32 k/255 can yield k-1.
- B8 InstructPix2Pix input passed as a tensor in [-1, 1] (edit_ip2p.py:19); the diffusers image processor treats a tensor without negative values as [0, 1] and rescales again (only when all pixels >= 128).
- B9 Edit LPIPS on GPU, defense LPIPS on CPU; LAION and MUSIQ-AVA via pyiqa on CPU; SigLIP google/siglip-base-patch16-224 with empty text, ArcFace insightface buffalo_l highest-score face, cells without a face excluded from ArcFace means.
- B10 Perturbation survival LPIPS(P(y), P(x)) / LPIPS(y, x); seven-purification averages pool images, prompts and purifications.
- B11 VAE-Trans validation: center crops, fixed noise seed 12345.
- B12 README.md:36 example command omits vae_trans from --purifications; configs include it.

README: numbers agree with configs and code; B1 is not described in README.
