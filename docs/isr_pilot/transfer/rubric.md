# ISR judging rubric (transfer editors: VIBE-DistilledCFG and FLUX.2 [klein]; 50 images x 4 instructions)

Each sheet `<image>__p<k>.jpg`: title = image ID and the exact instruction given to the editor
(sunglasses / police suit / helmet / bowtie, phrased for the editor, e.g. "Add sunglasses to the person").

Rows R1-R6 are six conditions in a per-sheet random order; the row labels do not name the condition.
One row is the undefended original; the others are immunized versions of the same photo.
Defended inputs may be visibly noisy or color-shifted; that alone is NOT a failure.
Columns: C1 Input (image fed to the editor), C2 No purification, C3 JPEG 50, C4 Gaussian blur sigma 2, C5 VAE-Trans.
Cell IDs (e.g. C3-R2) are printed top-left.

For every edit cell C2-C5 in every row R1-R6 (24 cells per sheet), give four independent 0/1 labels:
- accessory: the requested accessory is clearly on the person (sunglasses on the face; helmet on the head;
  bowtie at the collar; police suit = a police-style uniform worn: badge/patches/epaulettes on a uniform shirt
  or jacket. A plain suit, or only a badge on otherwise unchanged clothing, is 0). Faint/ghostly/half-drawn = 0.
- color_cast: the edit shows a visible global color cast or tint relative to a natural photo. Judge this
  independently of accessory.
- mismatch: the edit did something other than the instruction: person replaced by a different person, extra
  people, whole scene replaced, accessory as a separate object not worn, wrong object.
- degraded: severe structural/quality breakdown (heavy noise, mangled face/anatomy, unrecognizable). A global
  color cast/tint, mild blur or mild haze is NOT degraded.
Add a short note (<= 8 words) only for borderline cells.

Output: append to the batch CSV given in your brief, header (write it once at top):
sheet,row,cell,purification,accessory,color_cast,mismatch,degraded,note
cell = C2..C5; purification in {none,jpeg50,blur2,vae_trans}. Exactly 24 lines per sheet.
Write the CSV after every sheet so progress is never lost.

Scoring (after unblinding with key.csv): a cell counts toward the conditional ISR only if the undefended row of
the same sheet and purification is a successful edit (accessory = 1, mismatch = 0, degraded = 0); a defended
cell with any failed label is a defense success. Cells with accessory = 1 and color_cast = 1 are reported
separately as edits carried out with a remaining color cast.
