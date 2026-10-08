# Reproducibility update — 2026-10-08

The initial archive remains the snapshot described in README.md. This update adds browsable recovered sources, provenance reports, and selected main-experiment checkpoints under `reproducibility/`.

- [V1 recovery and checkpoint binding report](reproducibility/recovery_2026-10-08/README.md)
- [Figure/data/code map](reproducibility/traceability_2026-10-08/FIGURE_DATA_CODE_MAP.md)
- [Selected weight locator and SHA-256 list](reproducibility/recovery_2026-10-08/ARCHIVED_WEIGHT_LOCATORS.csv)
- [Figure-to-checkpoint bindings](reproducibility/recovery_2026-10-08/FIGURE_CHECKPOINT_BINDINGS.csv)

## Included weights

50 files: 10 V4 primary-150 containers with all six internal snapshots; 20 Place V5 confirmatory endpoint files (150 and 300 across ten seeds); and 20 Splitter V7.3 confirmatory endpoint files (150 and 300 across ten seeds). The other intermediate Place/Splitter checkpoints remain in the local inventory and are not duplicated here. Earlier development, debug, and obsolete model versions are listed for provenance but are not added as main-experiment weights.

The V4 file hashes match the Figure 2 controls and Figure 3 operation/gate records. Internal update-100 HDGM tensor fingerprints match all ten CA1 records. Place/Splitter seed, endpoint, and computational configurations match the recorded assays, but their historical result files do not save checkpoint SHA-256 values; historical byte identity remains unproven. The Figure 5 configuration name differs by the confirmatory suffix only.

## Evaluation entry point

From this repository root, preview the recovered V1 evaluation command:

```sh
python reproducibility/recovery_2026-10-08/run_figure3_v1.py
```

The wrapper defaults to the archived `weights/v4/` directory. It executes only with `--execute`; choose an empty `--output-dir` for a new evaluation. No intervention rerun, probe fitting, or core training was performed for this upload. Restoring a source file is not an end-to-end rerun claim. Original V1 evaluation-source SHA was not recorded in the historical JSON, so source semantics and the exact base-model SHA are the available identity evidence.

The original manuscript/data/figure bundle is still available in `hdgm_reproducibility_2026-09-19.zip`. Figure-map paths refer to its extracted root. Newly recovered Figure 5 plotting sources and title-cleanup provenance are under the traceability directory; `rebuild_figure5_capacity_boundary.py` remains unavailable, and full legacy artwork regeneration is not yet closed.

`UPLOAD_MANIFEST_2026-10-08.json` hashes the uploaded payload. Selected weight hashes are checked against the inventory before upload. The repository is intended to remain private. Reviewer access and public-release arrangements still require author confirmation; author contributions remain pending.

`ORIGINAL_RECOVERY_MANIFEST.json` preserves the pre-relocation local report hashes. The wrapper and README were adapted for repository-local weights; the upload manifest records their current bytes. Recovered historical V1 and base-model sources are unchanged.
