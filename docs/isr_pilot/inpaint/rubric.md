# Inpainting ISR judging rubric (SD Inpainting, 306 images x 4 instructions)

Each sheet `<image>__p<k>.jpg`: title = image ID and the exact instruction given to Stable Diffusion Inpainting:
"A {man/woman} in a hospital" (p0), "A {man/woman} in the gym" (p1), "A {man/woman} in a snowy street" (p2),
"A {man/woman} sitting next to a dog" (p3). The editor only repaints the BACKGROUND (everything outside the person);
the person is kept.

Rows R1-R7 are 7 conditions in a shuffled order; condition names are hidden. Columns: C1 Input (image fed to the
editor), C2 No purification, C3 JPEG 50, C4 Gaussian blur sigma 2, C5 VAE-Trans. Cell IDs (e.g. C3-R5) are printed
top-left. Inputs may be visibly color-shifted; color shift alone is NOT a failure.

Judge every edit cell C2-C5 in every row R1-R7 (28 cells per sheet) with three independent 0/1 labels:
- content: the requested content is clearly present in the repainted background.
  - hospital: recognizable medical setting or equipment (hospital bed, IV stand, monitor, wall medical equipment,
    hospital corridor, clinic room, medical staff). Plain white wall, only a sign/text, or a collage = 0.
  - gym: recognizable gym equipment or gym (dumbbells, barbell, rack, treadmill). Plain room/wall = 0.
  - snowy street: a recognizable outdoor snow scene (falling or lying snow outdoors). Street elements are NOT
    required. Only a snowflake pattern / texture without an outdoor snow scene = 0.
  - dog: a recognizable dog (head or body clearly visible). Blob, unrecognizable fragment, tiny cropped part = 0.
  - Faint, half-drawn, or needs careful searching = 0.
- mismatch: the edit did something other than the instruction: person replaced by a different person or the person
  region altered (thin seams at the mask edge do not count); extra people; for dog, a person or another animal
  instead of a dog; background replaced by an unrelated scene. The band of original background color around the
  person outline (mask dilation) is present in all conditions and does NOT count.
- degraded: severe structural/quality breakdown (heavy noise, background turned into unrecognizable texture or
  scribbles, large broken areas at the person/background boundary). Global color cast, mild blur or mild haze is
  NOT degraded. The mask-dilation band does NOT count.
Add a short note (<= 8 words) only for borderline cells.

Output: append to the batch CSV given in your brief; header (write it once at top):
sheet,cell,purification,content,mismatch,degraded,note
sheet = file name without extension; cell = e.g. C3-R5; purification in {none,jpeg50,blur2,vae_trans}
(C2=none, C3=jpeg50, C4=blur2, C5=vae_trans). Write the CSV after every sheet so progress is never lost.
Exactly 28 rows per sheet.
