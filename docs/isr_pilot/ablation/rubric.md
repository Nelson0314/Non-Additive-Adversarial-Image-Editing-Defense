# ISR judging rubric (ablation: ImmuLUT, B2, B3; ip2p, crosseditor100 x 4 instructions)

Each sheet `<image>__p<k>.jpg`: title = image ID and the exact instruction given to InstructPix2Pix
("Let the person wear sunglasses / a police suit / a helmet / a bowtie").

Rows R1-R4 are four conditions in a per-sheet random order; the row labels do not name the condition.
One row is the undefended original; the other three are color-mapped (immunized) versions of the same photo.
Columns: C1 Input (image fed to the editor), C2 No purification, C3 JPEG 50, C4 Gaussian blur sigma 2, C5 VAE-Trans.
Cell IDs (e.g. C3-R2) are printed top-left.

For every edit cell C2-C5 in every row R1-R4 (16 cells per sheet), give four independent 0/1 labels:
- accessory: the requested accessory is clearly on the person (sunglasses on the face; helmet on the head;
  bowtie at the collar; police suit = a police-style uniform worn: badge/patches/epaulettes on a uniform shirt
  or jacket. A plain suit, or only a badge on otherwise unchanged clothing, is 0). Faint/ghostly/half-drawn = 0.
- color_cast: the edit shows a visible global color cast or tint relative to a natural photo (for example a
  pink, orange, green or blue tint of skin, clothing or background). Judge this independently of accessory.
- mismatch: the edit did something other than the instruction: person replaced by a different person, extra
  people, whole scene replaced, accessory as a separate object not worn, wrong object.
- degraded: severe structural/quality breakdown (heavy noise, mangled face/anatomy, unrecognizable). A global
  color cast/tint, mild blur or mild haze is NOT degraded.
Add a short note (<= 8 words) only for borderline cells.

For every row, also rate the Input cell C1 once:
- naturalness: 1-5, how natural the input looks as an ordinary photograph (5 = indistinguishable from an
  unedited photo; 3 = noticeable but plausible color grading; 1 = clearly unnatural colors, patches or artifacts).

Output: append to the batch CSV given in your brief, header (write it once at top):
sheet,row,cell,purification,accessory,color_cast,mismatch,degraded,naturalness,note
- One line per edit cell (cell = C2..C5, purification in {none,jpeg50,blur2,vae_trans}, naturalness empty).
- One extra line per row with cell = C1, purification = input, the four 0/1 labels empty, and naturalness filled.
Exactly 20 lines per sheet. Write the CSV after every sheet so progress is never lost.

Scoring (after unblinding with key.csv): a cell counts toward the conditional ISR only if the undefended row
of the same sheet and purification is a successful edit (accessory = 1, mismatch = 0, degraded = 0).
A defended cell with accessory = 1 and color_cast = 1 is an edit that was carried out with a remaining color
cast, not a blocked edit; a cell with accessory = 0 is a blocked edit.
