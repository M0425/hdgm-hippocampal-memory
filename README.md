# HDGM hippocampal memory: private reproducibility archive

Working materials for the HDGM manuscript. This is a **private, pre-submission archive**, not a claim of public data availability. Author contributions remain pending confirmation. No software license has yet been selected.

## Contents and evidence map

| Manuscript item | Evidence | Primary archived material |
|---|---|---|
| Figure 1 | Conceptual functional architecture, not data | `manuscript/` image and `code/` |
| Figure 2 | Four-class acquisition and difficulty controls | `data/pythonProject32/fourclass_rapid_acquisition_v4_primary_results.json`, `figure2_missing_outputs/`, `figure_sources/figure2_E_rebalanced_v2_outputs/` |
| Figure 3 | Stage cuts, clamp controls, CA1 decoding | `data/New project/figure3_refined_v2_outputs/data/` |
| Figure 4 | Residual population maps, history-conditioned population separation | `data/New project/figure4_refined_outputs/data/` |
| Figure 5 | Aliasing/noise, frozen transfer probes and readouts | `data/pythonProject32/hdgm_difficulty_v2_1_frozen_outputs_confirmatory/` and Figure 5 provenance JSON |
| Figure A1 | Exploratory 3-seed structural controls and probe diagnostics | `pilot/checkpoint_diagnostics_v1/`, `pilot/factorial_3seed_outputs/`, `appendix_A1/` |
| Statistical audit | 464 numerical consistency checks | `tools/recheck_statistics.py`, `audit/`, generated `verification/` |

## Verification without training

Unzip the versioned archive into its own directory, then run:

```sh
python tools/verify_archive.py
python tools/recheck_statistics.py
```

The first command uses only the Python standard library. The second requires NumPy. `requirements-audit.txt` records versions available when packaging. Plotting Figure A1 additionally needs Matplotlib and Pillow; `python appendix_A1/build_figure.py` regenerates its outputs. No command above retrains a core model. Archived legacy training scripts may require local-path adaptation; they are preserved rather than presented as a tested universal training entry point.

## Scope and limitations

- Main assays use ten independently trained seeds per assay; the exploratory controls use three development seeds (11, 22, 33). Within-trial observations are not independent model replications.
- All 24 exploratory checkpoints (four variants, three seeds, two endpoints) are archived. The complete checkpoint sets for all main assays are not included. Source versions are hash-recorded snapshots; their presence alone does not prove identity with every historical training run.
- Update 150 is the original representation endpoint; update 300 probe analyses are post hoc. Every variant retains predictive recurrence. The no-memory variant disables the online associative matrix, not all memory.
- Pooled and position-conditioned probes differ in sampling and aggregate capacity. These results do not establish absence of information, architecture necessity, cross-position transfer, or a fit to recorded neural firing rates.
- Figure 5 uses the V2.1 frozen confirmatory results, not the older V2 results. Its final PDF, numerical inputs, provenance, and available scripts are archived. The legacy integrated script depends on an unavailable external `rebuild_figure5_capacity_boundary.py`; complete artwork regeneration is therefore **not yet closed**.
- Figure 1 is an illustrative diagram. Statistical consistency checks verify saved quantities, not independent biological validity or a successful end-to-end retraining.
- The manuscript contains a pending access statement and a blank Author Contributions section. These require author review before submission. Private GitHub storage does not itself give reviewers access.

`PROVENANCE.json` maps copied artifacts to original local sources. `MANIFEST.json` records SHA-256 checksums for the packaged files. `audit/` contains prior audit records; `verification/` contains the rerun in the portable layout. Avoid loading untrusted PyTorch checkpoints; the audit verifies them as byte files and does not deserialize them.
