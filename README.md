# Figure5 DEF incremental update — 2026-10-16

This folder adds newly reconstructed d/e/f plotting scripts to the existing repository.
It does not replace or require re-uploading the September archive or October8 checkpoints.
Two compact archived JSON inputs are included so the new scripts run independently of
the layout inside the older ZIP. Their numerical contents are unchanged.

From repository root:
```
python reproducibility/figure5_DEF_reconstruction_2026-10-16/rebuild_figure5_panel_d.py
python reproducibility/figure5_DEF_reconstruction_2026-10-16/rebuild_figure5_panel_e.py
python reproducibility/figure5_DEF_reconstruction_2026-10-16/rebuild_figure5_panel_f.py
```
Requires NumPy, Matplotlib, Pillow; saved-data processing only, no model training.
Outputs go into reconstructed_outputs/ next to the scripts.
Three scripts were executed and checked in the complete local package before preparation
of this increment. Paths were adapted to this self-contained layout.

Important: these are reconstructed implementations, not recovered historical scripts.
Original bootstrap random streams are unknown. E displays the explicitly archived
interval; newly computed intervals are exported separately. The manuscript figure
is not replaced. See RECONSTRUCTION_REPORT.json and the original saved provenance.

Upload the reproducibility folder INSIDE HDGM_github_increment_2026-10-16 at the repository
root. Do not upload this outer packaging directory or ZIP itself. Existing files in
reproducibility remain; a new dated subdirectory is added. Full 10-batch re-upload is
unnecessary for this plotting-code update.
