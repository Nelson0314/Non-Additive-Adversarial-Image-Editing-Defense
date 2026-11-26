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
