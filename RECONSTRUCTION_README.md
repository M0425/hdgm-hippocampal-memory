# Figure5 d/e/f reconstruction

These scripts were newly written on 2026-10-16, not recovered from the original run.
They read the archived ten-seed V2.1 frozen results using package-relative paths.

```
python rebuild_figure5_panel_d.py
python rebuild_figure5_panel_e.py
python rebuild_figure5_panel_f.py
```

Outputs: reconstructed_outputs alongside these scripts (PDF, SVG, PNG, seed CSV and statistics JSON).
No training, decoder fitting or intervention evaluation is invoked.
Shared definitions are in reconstructed_common.py.

- d: HDGM/PAM full-minus-reset accessibility; normalized access AUC; the four-test
  Holm family includes two accessibility and two stability contrasts. Both access
  adjusted p values are 0.0078125, matching the archive. 16,000 mean-bootstrap draws.
- e: seven unique conditions per seed; the noise0.05 baseline is counted through
  alias0 only. Center within seed, then calculate the pooled descriptive correlation.
  r matches the archive exactly. Resample 10 seed clusters with replacement 12,000
  times, keeping all seven conditions within each sampled seed.
- f: standardized mean over sample SD across seeds. Metrics are full-minus-reset
  decoding, full-minus-reset separation ratio, and choice-minus-0.5. Exact 1,024
  sign flips for each cell; one 24-test Holm family. All 24 effect sizes and all 24
  adjusted p values match the saved provenance within 1e-12.

## Historical vs reconstructed uncertainty
The original bootstrap random seed/stream was not saved. We use 20261016 for new
draws and do NOT claim identical old intervals. In panel e the displayed interval
is explicitly labeled ARCHIVED and read from the original provenance JSON; its
source is hash-recorded, and r is independently checked first. The new interval
is exported separately. The shaded slope band is newly reconstructed. In panel d
mean intervals are newly recomputed. Density bandwidth and layout are documented
visual choices rather than recovered original code.

These are reproducibility reconstructions, not drop-in pixel-identical replacements
for the manuscript panels. The existing reference figure5.pdf remains unchanged.
The missing historical whole-layout module is still not recovered; this addition
restores data-driven scripts for the previously missing d/e/f panels.

Current GitHub layout: scripts and both input JSONs are in the repository root; reconstructed_common.py supports this flat layout and the original data/ layout.
