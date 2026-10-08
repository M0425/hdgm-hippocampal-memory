# Figure 5. Structured-memory capacity under alias corruption and observation noise

All quantitative panels use the ten frozen V2.1 confirmatory seed records
(101, 202, 303, 404, 505, 606, 707, 808, 909, 1010). Alias corruption and
observation noise are independent one-dimensional scans, not a factorial
perturbation grid. Alias α = 0 and noise σ = .05 denote the same baseline.

**A.** Conceptual comparison of structural aliasing and observation noise at
a frozen checkpoint. EC–CA3–CA1 labels illustrate the computational route;
the diagram does not display measured neuron-level activity.

**B.** The accepted population-bifurcation portrait, plotted in history
decoding versus behavioral accuracy coordinates. Centers are condition means;
fine lines connect observations from the same seed. The main axes and marginal
distributions retain the zoomed display ranges of the accepted B; not all raw
seed endpoints fall inside the view. The ten internal marks are seed-count
glyphs, not neuron positions or raw sample coordinates. Envelope size encodes
relative bivariate mean-uncertainty magnitude, not a spatial confidence region;
violet halo size is based on history decoding. Trunk width encodes mean
split-half stability. Basins, attractor and manifold labels are interpretive,
not fitted dynamical potentials or evidence of a formal bifurcation.

**C.** Seed-by-condition behavioral accuracy matrices for HDGM, TEM and
stateless models, with the same seed order and common absolute 0–1 color scale.
Points below show means and unadjusted two-sided 95% Student-t confidence
intervals across ten seeds (df = 9). HDGM uses a 0.40–0.82 y range;
TEM and stateless share a clearly labeled 0.47–0.56 zoom. Dashed lines mark
chance accuracy (0.50). The repeated baseline is shown in both scan blocks
but is not pooled as an extra independent observation.

**D.** History access (full-history transfer decoding minus history-reset
transfer decoding). Open circles are HDGM seeds and squares are TEM seeds;
diamonds and intervals are HDGM means and 95% percentile bootstrap intervals
(16,000 resamples). Ridges use a fixed Gaussian bandwidth of 0.034 and are
normalized to unit peak separately; ridge heights are not sample counts.
HDGM–TEM AUC Holm-adjusted p values are taken from the frozen summary.

**E.** Descriptive association between seed-centered history access and
choice accuracy, using seven unique conditions per seed (70 observations)
after removing the duplicated baseline. The line is an ordinary least-squares
fit; the band and correlation interval use 12,000 seed-cluster bootstrap
resamples. The displayed association is not a causal or mediation estimate.

**F.** Paired standardized effects (d_z) for history retained, branch formed
and decision successful. Each cell contains ten seed marks. Exact sign-flip
tests are Holm-adjusted across the 24 cells; dark borders and asterisks denote
adjusted p < .05. Vertical links organize readouts and do not assert mediation.

## Production notes

- Original A/D/E/F numerical calculations are retained from the supplied script.
- New B and C are rendered natively into the figure, not pasted as PNG images.
- C's 24 means and 48 interval endpoints are programmatically checked against
  the previously accepted C summary CSV.
- Full master dimensions: 431.8 × 368.3 mm; minimum text: 8 pt; PNG: 360 dpi.
  This is an editable full-size master, not a claim of compliance with a
  particular journal's final column-width requirements.
- SVG and PDF preserve vector geometry; SVG text remains editable.
- The separate provenance JSON records data/script checksums and statistics.
