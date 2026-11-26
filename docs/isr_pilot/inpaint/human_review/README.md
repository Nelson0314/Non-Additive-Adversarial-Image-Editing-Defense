# Human review of the SD Inpainting ISR judgments

Sample: 60 cells of docs/isr_pilot/inpaint/judgments.csv, drawn with seed 20260812 among cells whose Undefended edit
Claude Sonnet judged successful; 10 cells per defense condition, of which 5 judged defense successes and 5 judged
defense failures; purifications rotated. The review page shows the instruction, the purification, the input, the
Undefended edit of the same purification and the edit under review, without condition names or Sonnet labels.

Labels: the reviewer first states whether the Undefended edit carries out the instruction; cells where it does not
are excluded from the comparison. For the remaining cells the reviewer labels content, mismatch and degraded.

Files
- answers.csv: first pass of the review.
- audit.csv: final review labels. Three cells (18, 56, 58) are additionally excluded because their Undefended
  background is only blurred or shows no clear scene cue, the standard applied to the other excluded cells.
- agreement.txt, agreement_audit.txt: agreement with Sonnet for answers.csv and audit.csv
  (../measure_inpaint_human_agreement.py, run with python3 on the review key).

Disagreements (audit.csv): Sonnet labels the Undefended edit successful when the background shows only weak cues
(blurred or darkened background, a single small device, no snow), and labels defended edits as failures when the
scene cue is weak or at the image border, when scene-consistent people are added, or when the mask-dilation band
is visible. Both directions raise the ISR of every condition.
