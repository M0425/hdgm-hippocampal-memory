# HDGM research reproducibility materials

**A Hippocampal-Inspired Dynamic Gating Model: Linking Selective Memory Operations to History-Dependent Population Representations**

This repository contains the archived study materials and a later Figure5 plotting reconstruction.

## Repository contents
- `hdgm_reproducibility_2026-09-19.zip`: original archived experiment code, data, figures and exploratory controls. Unzip into a separate folder before using its internal instructions.
- `reproducibility/recovery_2026-10-08/`: recovered source code and archived main checkpoints.
- `reproducibility/traceability_2026-10-08/`: source and figure traceability materials.
- `REPRODUCIBILITY_UPDATE_2026-10-08.md`: earlier recovery/update documentation.
- Root-level `rebuild_figure5_panel_d.py`, `rebuild_figure5_panel_e.py`, `rebuild_figure5_panel_f.py`: newly reconstructed saved-data plotting scripts, dated 2026-10-16.

## Figure5 d/e/f
The current repository places these scripts and their two JSON inputs at the root.
Run from that directory:
```
python rebuild_figure5_panel_d.py
python rebuild_figure5_panel_e.py
python rebuild_figure5_panel_f.py
```
Requires NumPy, Matplotlib and Pillow. Output goes into `reconstructed_outputs/`.
The shared module also accepts the original `data/` input subdirectory layout.
No model training or probe fitting is performed by these commands.

The scripts are reconstructed implementations, not recovered historical sources.
D AUC adjusted p values, E correlation and all24 F effect sizes/adjusted p values
were checked against saved results. Original bootstrap random streams were not
recorded; archived and newly recomputed intervals are explicitly distinguished.
See `RECONSTRUCTION_README.md` and `RECONSTRUCTION_REPORT.json`.

`MANIFEST.csv` describes only the17-file Figure5 reconstruction addition, not the
entire historical repository. Older upload manifests describe their dated collections.

This repository does not claim full end-to-end retraining has been rerun in a new
environment. Environment versions for reconstruction are distinct from historical
training versions. The author-designed AI-rendered Figure1 has separate records.
No new licence is granted by this README.
