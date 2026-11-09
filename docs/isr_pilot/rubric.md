# ISR judging rubric (ip2p, 100 images x 4 instructions)

Each sheet `<image>__p<k>.jpg`: title = image ID and the exact instruction given to InstructPix2Pix
("Let the person wear sunglasses / a police suit / a helmet / a bowtie").

Rows (condition): R1 Undefended, R2 PhotoGuard, R3 DCT-Shield, R4 BlurGuard, R5 SIFM, R6 Mist, R7 CAPGuard 2.5x.
Columns: C1 Input (image fed to the editor; R1 = original photo). C2 No purification, C3 JPEG 50,
C4 Gaussian blur sigma 2, C5 VAE-Trans. Cell IDs (e.g. C3-R7) are printed top-left.
Defended inputs may be visibly color-shifted (especially R7); that alone is NOT a failure.

Judge every edit cell C2-C5 in every row R1-R7 (28 cells per sheet) with three independent 0/1 labels:
- accessory: the requested accessory is clearly on the person (sunglasses on the face; helmet on the head;
  bowtie at the collar; police suit = a police-style uniform worn: badge/patches/epaulettes on a uniform shirt
  or jacket. A plain suit, or only a badge on otherwise unchanged clothing, is 0). Faint/ghostly/half-drawn = 0.
- mismatch: the edit did something other than the instruction: person replaced by a different person, extra
  people, whole scene replaced, accessory as a separate object not worn, wrong object.
- degraded: severe structural/quality breakdown (heavy noise, mangled face/anatomy, unrecognizable). A global
  color cast/tint, mild blur or mild haze is NOT degraded.
Add a short note (<= 8 words) only for borderline cells.

Output: append to the batch CSV given in your brief, header (write it once at top):
sheet,cell,condition,purification,accessory,mismatch,degraded,note
condition in {Undefended,PhotoGuard,DCT-Shield,BlurGuard,SIFM,Mist,CAPGuard}; purification in {none,jpeg50,blur2,vae_trans}.
Write the CSV after every sheet so progress is never lost. Exactly 28 rows per sheet.
