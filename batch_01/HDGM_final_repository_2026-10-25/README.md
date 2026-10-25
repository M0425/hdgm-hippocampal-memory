# HDGM final reproducibility materials

Manuscript: Selective Memory Operations Shape History-Dependent Population Representations and Choice under Sensory Ambiguity: A Hippocampal-Inspired Dynamic Gating Model

This directory is the final entry point for the manuscript's experiment code, saved outputs, checkpoints, and plotting tools. Earlier dated archives provide provenance; use this directory for the final assay scripts.

## Start here

1. Install dependencies: `python -m pip install -r requirements.txt`.
2. Verify the archive: `python tools/verify_package.py`.
3. Recalculate saved-output statistics: `python tools/recheck_statistics.py`.
4. Regenerate plots using the commands in RUN_COMMANDS.md.

EXPERIMENT_MAP.csv maps manuscript analyses to scripts. AUTHOR_CODE_COMPARISON.csv maps the author's ten supplied files to packaged scripts. The final CA1 validation is V6.2 (`experiments/ca1_validation.py`), identical to the author's `CA1终.py`; V6.1 `CA1.py` is superseded.

## Contents

- experiments/: selected final experiment entry points.
- data/: saved simulation and analysis outputs.
- weights/: main experimental checkpoints; pilot/: exploratory structural controls.
- plotting/: saved-data plotting scripts; tools/plot.py is the portable plotting entry point.
- reference_figures/: reference figure assets retained from the previous package.
- figure1_design/: retained conceptual illustration design records.
- configs/, environment/, provenance/: configuration and source records.

## Naming and reproducibility

Historical TEM/TEMLite code identifiers denote the predictive-associative comparator (PAM) used in these assays, not the full published TEM model. The plot runner updates displayed TEM labels to PAM and panel letters to lowercase without changing numerical data.

Figure 5 D/E/F scripts were reconstructed from saved numerical outputs; see plotting/figure5/RECONSTRUCTION_README.md. The original whole-layout assembly script is unavailable. Figure 1 is conceptual artwork with retained design records. Reference figure assets are not evidence of pixel-identical reproduction by these scripts.

Training scripts preserve original scientific configurations and may need checkpoint/output path configuration; see RUN_COMMANDS.md. Dependencies are listed, but historical training versions are not fully locked; see environment/README.md. This release does not claim a new end-to-end retraining run. Verification performed during packaging is recorded in PACKAGING_CHECKS.json.

No software license has been assigned in this archive.
