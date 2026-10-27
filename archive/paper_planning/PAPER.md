<!--
CVPR 2027 submission draft (content only; paste into the official CVPR 2027 LaTeX template on Overleaf).
Conventions:
  [TODO: ...]   experiment or text not yet available
  [NUM]         number to be filled from the final run
  \cite{key}    citation key; keys follow the literature table in docs/research_tree.html
Page budget (main paper, 8 pages excluding references): Intro 1.25, Related 0.75, Method 2.0, Experiments 3.5, Discussion 0.5.
Status of each block is recorded in docs/pseudo_paper.html.
-->

# Color-Mapping Immunization: Purification-Robust Protection of Images against Diffusion-Based Editing

*Anonymous CVPR 2027 submission — Paper ID [NUM]*

## Abstract

Instruction-guided diffusion editors allow any published portrait to be rewritten with a single sentence. Image immunization counters this by perturbing an image before release so that subsequent edits deviate from the attacker's intent. Existing immunization methods optimize bounded additive perturbations pixel by pixel with projected gradient descent (PGD); such perturbations are largely removed by common purifications such as JPEG compression, blurring, resizing, and VAE reconstruction. We observe that, on the plane spanned by defense distortion and edit displacement, existing methods lie close to a single line of high correlation, so that protection strength is mainly bought by the distortion budget; what separates methods is how much of the displacement survives purification. Following the practice of photographic color grading, we move the perturbation from pixels to the space of global color mappings: each image is protected by an optimized 3D lookup table (LUT) that maximizes the latent displacement of the editor's VAE encoder without access to any editing instruction, under an image-adaptive feasible set that encodes perceptual properties of the source image. At matched defense distortion against [NUM] baselines, our method retains [NUM]% of its edit displacement across seven purifications (baselines: [NUM]–[NUM]%) and [NUM]× the baseline displacement after VAE-based purification, with consistent trends on [TODO: model list]. A perceptual evaluation tailored to tone changes, including a user study, shows [TODO]. We further show that a single universal LUT optimized over many images provides image-agnostic protection.

## 1. Introduction

<!-- P1: problem -->
Public instruction-guided editors \cite{brooks2023instructpix2pix, labs2025flux1kontext} make it possible to alter a person's appearance, clothing, or context in a published photograph with a short text prompt. Image immunization \cite{salman2023photoguard} protects an image before release: a perturbation is applied so that an editor operating on the protected image produces results that deviate from those it would produce on the original.

<!-- P2: common form of existing methods -->
Almost all existing immunization methods share one form. They optimize an additive perturbation $\delta$ under an $\ell_\infty$ or $\ell_2$ budget with PGD, targeting the editor's encoder, denoiser, or attention maps \cite{salman2023photoguard, liang2023mist, zhang2023editshield, dia2025, dctshield2025}. Because $\delta$ is set independently at each pixel, the defense is a spatial pattern attached to the pixel grid of the original image.

<!-- P3: purification -->
Common image processing rewrites exactly this pattern. JPEG compression, Gaussian blur, crop-and-resize, and VAE reconstruction \cite{sandoval2023jpeg, honig2025noisyupscaling, purifyonce2026} strip most of the perturbation while leaving the photograph intact. Under matched distortion, the edit displacement of eleven existing methods drops from [NUM] to [NUM] after seven standard purifications and to [NUM] after VAE-based purification (Sec. 4.4).

<!-- P4: observation -->
Comparing methods is further complicated by their budgets. At native settings, the ranking of methods by edit displacement nearly coincides with their ranking by defense distortion, and on the distortion–displacement plane all methods fall close to one line (Fig. 4). Increasing the budget moves a method along this line. We therefore compare methods at matched distortion, where the question becomes which perturbation carrier survives purification.

<!-- P5: idea -->
Photographers routinely change the colors and tones of an entire image with lookup tables, and viewers accept such changes as a style. A LUT maps each pixel by its color alone and introduces no spatial structure. We use the LUT as the immunization carrier: the defense is a per-image color mapping optimized against the editor, which lives in a low-dimensional, globally applied parameter space. Purifications that preserve image content also largely preserve its color distribution, and with it the defense.

<!-- P6: challenges -->
Two problems arise. First, global color changes can be large and must remain natural; we constrain the LUT with an image-adaptive feasible set in which each constraint corresponds to a perceptual property of the source image, and solve with projection. Second, reference-based distortion metrics count any change of tone as degradation; we complement them with color-aligned distortion, preference models, and a user study.

<!-- P7: contributions -->
Our contributions are:
- We show that existing immunization methods lie on a common distortion–displacement line and that their protection is largely removed by purification, motivating comparison at matched distortion after purification.
- We propose color-mapping immunization, which optimizes a per-image 3D LUT against the editor's encoder without editing instructions, under an image-adaptive feasible set solved by projection.
- At matched distortion, the method retains substantially more displacement after seven purifications and VAE-based purification than eleven additive baselines, and the trend holds across [TODO: models and editing scenarios].
- We introduce an evaluation of naturalness for tone-changing defenses, including color-aligned distortion and a user study, and a universal LUT that protects unseen images.

## 2. Related Work

**Immunization against diffusion editing.** PhotoGuard \cite{salman2023photoguard} introduced encoder and diffusion attacks for protecting images from Stable Diffusion editing. Subsequent methods change the target or the budget: Mist and AdvDM \cite{liang2023mist, liang2023advdm} combine textural and semantic losses; EditShield \cite{zhang2023editshield} targets instruction-guided editing; DIA \cite{dia2025} attacks inversion trajectories for inpainting; DCT-Shield \cite{dctshield2025} places perturbations in the JPEG quantization domain; FaceLock \cite{wang2025facelock} targets facial identity; DiffVax \cite{diffvax2026} trains a feed-forward immunizer; BlurGuard \cite{blurguard2025} shapes perturbations to survive purification. [TODO: SIFM, DANP, DAYN.] All of them apply additive pixel perturbations; recent work on DiT editors \cite{decontext2025, nulledit2026} keeps this form.

**Purification and breaking immunization.** Simple transformations such as JPEG \cite{sandoval2023jpeg} and noisy upscaling \cite{honig2025noisyupscaling} remove protective perturbations. Purify Once, Edit Freely \cite{purifyonce2026} shows that VAE-based purifiers break six immunization methods without access to the protected images. CAT \cite{cat2025} attributes the effectiveness of protective perturbations to latent distortion and removes it with adapter fine-tuning. Evaluations question whether protected images actually block edits \cite{icip2025disruptive, doprotect2026}.

**Color-based adversarial examples.** Unrestricted adversarial examples change color instead of adding noise: hue and saturation shifts \cite{hosseini2018semantic}, per-image color lookup tables \cite{laidlaw2019recoloradv}, piecewise-linear color filters \cite{zhao2020ace, zhao2023advcf}, semantic recoloring \cite{shamsabadi2020colorfool, yuan2022ncf}, and colorization \cite{bhattad2020cadv}. These target classifier misprediction. We adopt the LUT carrier for editing immunization and study its behavior under purification.

**Perceptual evaluation of color changes.** Full-reference metrics such as LPIPS and DISTS \cite{zhang2018lpips, ding2020dists} and color-difference formulas \cite{sharma2005ciede2000} penalize global tone changes. Color-specific metrics \cite{zhang1997scielab, rajashekar2010color, chen2023cdflow}, human preference models \cite{kirstain2023pickscore, ma2025hpsv3}, and human studies of unrestricted adversarial examples \cite{scooter2025} provide complementary views. [TODO: position our color-aligned distortion.]

## 3. Method

### 3.1 Threat model

The defender processes an image $x$ before release and outputs $\tilde{x}$. The attacker edits $\tilde{x}$ with a public editor $E(\cdot, p)$ under an unknown instruction $p$, optionally after a purification $P$. We assume white-box access to the editor's weights, no modification of the attacker's model, and no knowledge of $p$: the defender uses only an empty prompt or a class word. The defense succeeds when $E(P(\tilde{x}), p)$ deviates from $E(P(x), p)$, measured by LPIPS between the two edits (edit displacement).

### 3.2 Color-mapping carrier

A 3D LUT $T_\theta: [0,1]^3 \to [0,1]^3$ with $33^3$ lattice nodes maps each pixel by trilinear interpolation, $\tilde{x}_i = T_\theta(x_i)$. Pixels of equal color receive equal outputs, so the defense contains no spatial pattern. [TODO: short analysis contrasting how purification operators act on $T_\theta(x)$ versus $x + \delta$.]

### 3.3 Objective

We maximize the displacement of the editor's VAE encoder latent,
$$\max_\theta \; \| \mathcal{E}(T_\theta(x)) - \mathcal{E}(x) \|_2^2,$$
which requires one forward pass and no instruction. Alternative objectives are compared in Sec. 4.8.

### 3.4 Image-adaptive feasible set

Instead of a sum of independent penalties, we constrain $T_\theta$ to a feasible set $\mathcal{C}(x)$ computed from the source image and enforce it with an ADMM projection after each gradient step; only a lower bound on defense distortion remains in the loss. Each constraint corresponds to a perceptual property:
- color-space Lipschitz continuity of the mapping, which prevents color fringes in neutral and white regions;
- a per-pixel skin ellipse in $(a^*, b^*)$ under the skin covariance, which preserves the memory color of skin;
- a per-tile covariance bound, which keeps local color consistent;
- [TODO: higher-level formulation grounded in perceptual properties of photographs (tone order, memory colors, colorfulness, contrast) and calibrated on common color-grading presets].

With projection, no multiplier is active at delivery and the number of feasible checkpoints rises from 1/180 to 105–165/180 [prelim, 2 images].

### 3.5 Strength control

A lower bound on defense LPIPS, linearly ramped over the first 600 steps, sets the operating point, playing the role of the budget $\epsilon$ in additive methods.

### 3.6 Universal LUT

[TODO] A single $T_\theta$ is optimized over a training set of images and applied to unseen images. A variant targets a category of instructions; because it uses instructions during training, it is evaluated under a separate threat model.

## 4. Experiments

### 4.1 Setup

**Data.** Eight portraits (512²) with subject masks; [TODO: Pexels portraits, N = 306–500].
**Attack.** InstructPix2Pix with Euler-Ancestral sampling (50 steps, text guidance 7.5, image guidance 1.8, seed 20260812) and four accessory instructions from FaceLock \cite{wang2025facelock} (8 × 4 = 32 edits).
**Purification.** JPEG (quality 80/50/30), Gaussian blur (σ = 1, 2), crop-resize (10%), rotation (15°), and VAE-Trans \cite{purifyonce2026}.
**Baselines.** Eleven additive methods, each rescaled to a common defense LPIPS (matched-distortion arm) and at native budgets; DiffVax \cite{diffvax2026} has no budget parameter and is reported separately; BlurGuard \cite{blurguard2025} serves as a purification-aware additive control.
**Metrics.** Edit displacement (LPIPS between undefended and defended edits; full image, subject, background), retention (purified / unpurified displacement), semantic and identity scores, and defense distortion (LPIPS, DISTS, ΔE00, PSNR, SSIM, FSIM, VMAF).

### 4.2 Defense without purification and the distortion–displacement line

Table 1 compares methods at matched distortion. [Prelim: ours 0.287 defense LPIPS, 0.341 edit displacement; baselines at 0.30–0.37 defense LPIPS reach 0.24–0.39.] Fig. 4 plots edit displacement against defense distortion for budget sweeps of every method. Global color mappings follow edit LPIPS ≈ 0.99 × defense LPIPS + 0.064 over 136 conditions [TODO: overlay additive sweeps; repeat with DISTS and PieAPP as the distortion axis]. The lower bound on distortion moves our method along the line (0.287 → 0.326 → 0.409).

| Method | Defense LPIPS | DISTS | ΔE00 | PSNR | Edit LPIPS (full / subject / bg) | CLIP-S | ArcFace |
|---|---|---|---|---|---|---|---|
| [11 baselines, matched] | | | | | | | |
| Ours | | | | | | | |

*Table 1. Matched-distortion comparison on ip2p.* [TODO: fill from baseline/results/aligned and color trial results.]

### 4.3 What purification removes

[TODO] For each purification operator $P$ we measure how much of the defense survives, $\mathrm{LPIPS}(P(\tilde{x}), P(x)) / \mathrm{LPIPS}(\tilde{x}, x)$ (Fig. 2), next to the retention of edit displacement, and sweep purification strength (JPEG quality, blur σ, resize ratio) to obtain displacement curves (Fig. 5). [TODO: discuss BlurGuard, whose displacement after VAE-Trans (0.191) is close to ours.]

### 4.4 Robustness to purification

| Method | Defense LPIPS | No purification | 7-purification mean | Worst | Retention | VAE-Trans |
|---|---|---|---|---|---|---|
| Baselines (11, matched) | 0.30–0.37 | 0.24–0.39 | 0.12–0.24 | [NUM] | [NUM] | 0.02–0.07 |
| Random LUT (matched) | 0.207 | 0.290 | 0.299 | 0.233 | 1.04 | 0.196 |
| Ours | 0.287 | 0.341 | 0.302 | [NUM] | 0.892 | 0.185 |

*Table 2. Edit displacement (LPIPS) after purification; 8 images × 4 instructions, ip2p, primary seed. Preliminary numbers.*

A random LUT at matched distortion survives purification to a similar degree, indicating that robustness is a property of the carrier, while optimization raises displacement by 32–61% over random LUTs of the same family.

### 4.5 Generalization across models

**Transfer.** Defenses computed on ip2p and evaluated with FLUX.1 Kontext \cite{labs2025flux1kontext} and UltraEdit \cite{zhao2024ultraedit} preserve the method ranking (Spearman 0.951 with ip2p).
**Per-model optimization.** [TODO] Ours and four representative baselines optimized and evaluated on SD 2.1-base, SD inpainting, and FLUX.1 Kontext. Our objective needs only the editor's VAE encoder, so optimization on FLUX does not back-propagate through its transformer; baselines whose losses require denoiser gradients are compared only where they can be ported with their own losses.

| Model | Method | Defense LPIPS | No purification | 7-purification mean | VAE-Trans |
|---|---|---|---|---|---|
| [TODO] | | | | | |

*Table 3. Generalization across editors.*

### 4.6 Perceptual quality of tone-changing defenses

Reference-based metrics (PieAPP, TOPIQ-FR, AHIQ, CKDN, CW-SSIM, NLPD, GMSD, MS-SSIM) and PickScore rank color-mapping defenses below additive ones, since they score any tone change as a difference from the reference. [TODO] We therefore report (i) color-aligned distortion, computed after fitting a global color transform from the defended image back to the original; (ii) the same metrics on common color-grading presets as a reference band of accepted edits; (iii) color-difference \cite{chen2023cdflow}, no-reference aesthetic, and preference \cite{ma2025hpsv3} scores; and (iv) a user study with two-alternative naturalness judgments and edited/unedited detection following \cite{scooter2025}.

| Method | Color-aligned LPIPS | Preset band | CD-Flow | Aesthetic | HPSv3 | User study |
|---|---|---|---|---|---|---|
| [TODO] | | | | | | |

*Table 4. Perceptual quality.*

### 4.7 Universal LUT

[TODO] Displacement and retention of a universal LUT on unseen images, against per-image LUTs and baselines; instruction-category variants.

### 4.8 Ablations

| Axis | Variants | Question |
|---|---|---|
| Constraint formulation | penalties in the loss (augmented Lagrangian) / feasible set with ADMM projection / feasible set with principled constraints | conflict between constraints, feasibility, appearance, displacement |
| Feasible-set components | + Lipschitz, + skin ellipse, + tile covariance | which defect each constraint removes |
| Strength | distortion floor 0.27 / 0.31 / 0.37 | movement along the line; retention |
| Carrier | RGB curves / (a*, b*) warp / 3D LUT 17³, 33³ | capacity versus attainable distortion |
| Objective | encoder latent / equivariance residual / implicit classifier | objective sensitivity (within 3%) |
| Random control | random LUT, random filter at matched distortion | carrier versus optimization |

*Table 6. Ablation axes.* [TODO: numbers.]

## 5. Discussion and Limitations

Edit displacement and edit failure are distinct: at the matched-distortion point, in 19 of 32 cells no method prevents the instructed accessory from appearing. Adaptive attackers that estimate and invert a global color mapping, including image-adaptive LUT enhancement networks \cite{zeng2020lut} and color normalization, are not evaluated. Tone changes remain visible by design; the perceptual evaluation in Sec. 4.6 measures their acceptability. Per-image optimization takes 12–25 minutes per image on an RTX 3090, which the universal LUT partly addresses.

## 6. Conclusion

[TODO]

## Supplementary Material (outline)

- A. Protocol details: data, masks, prompts, purification parameters, seeds, matched-distortion rescaling.
- B. Baseline provenance and budget conversion.
- C. Per-image results and appearance sheets.
- D. Evaluation validity: cells where undefended edits fail, blocking rates.
- E. Feasible-set derivation and projection algorithm.
- F. User-study interface and statistics.
